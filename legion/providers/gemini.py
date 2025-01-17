"""Google's Gemini-specific implementation of the LLM interface"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Type
from unittest.mock import MagicMock

from pydantic import BaseModel

from ..errors import ProviderError
from ..interface.base import LLMInterface
from ..interface.schemas import Message, ModelResponse, ProviderConfig, Role, TokenUsage
from ..interface.tools import BaseTool
from . import ProviderFactory


class GeminiFactory(ProviderFactory):
    """Factory for creating Gemini providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new Gemini provider instance"""
        # If no config provided, create one with defaults
        if config is None:
            config = ProviderConfig()

        # If no API key in config, try to get from environment
        if not config.api_key:
            config.api_key = os.getenv("GEMINI_API_KEY")

        # Set default base URL if not provided
        if not config.base_url:
            config.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        return GeminiProvider(config=config, **kwargs)


class GeminiProvider(LLMInterface):
    """Google's Gemini-specific provider implementation"""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    SUPPORTED_MODELS = {
        "gemini-1.5-flash-latest": "gemini-1.5-flash",
        "gemini-1.5-pro-latest": "gemini-1.5-pro",
        "gemini-pro": "gemini-pro",
        "gemini-2.0-flash-exp": "gemini-2.0-flash-exp"
    }

    def _setup_client(self) -> None:
        """Set up the OpenAI client with the Gemini API key and base URL."""
        # Check config first, then environment
        api_key = self.config.api_key if self.config else None
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ProviderError("API key is required")

        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.DEFAULT_BASE_URL,
            timeout=60,
            max_retries=3
        )

    async def _asetup_client(self) -> None:
        """Initialize async Gemini client"""
        self._setup_client()  # Reuse sync setup since AsyncOpenAI handles both

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Gemini API."""
        formatted_messages = []
        for message in messages:
            formatted = {"role": message.role.value}

            # Ensure content is never None/empty for Gemini
            formatted["content"] = message.content if message.content else " "  # Use space instead of empty string

            if message.tool_calls:
                formatted["tool_calls"] = []
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
                        formatted["tool_calls"].append(tool_call)
                    else:
                        formatted["tool_calls"].append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments
                            }
                        })

            if message.role == Role.TOOL:
                if message.name:
                    formatted["name"] = message.name
                if message.tool_call_id:
                    formatted["tool_call_id"] = message.tool_call_id

            formatted_messages.append(formatted)

        return formatted_messages

    def _extract_content(self, response: Any) -> str:
        """Extract content from response."""
        try:
            message = response.choices[0].message
            if message.content:
                return message.content
            elif hasattr(message, "tool_calls") and message.tool_calls:
                # If no content but has tool calls, return a descriptive message
                tool_names = [tc.function.name for tc in message.tool_calls]
                return f"Using tools: {', '.join(tool_names)}"
            return ""
        except Exception:
            return ""

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from response."""
        try:
            message = response.choices[0].message
            if not hasattr(message, "tool_calls") or not message.tool_calls:
                return None

            tool_calls = []
            for tool_call in message.tool_calls:
                # Handle both MagicMock and actual response objects
                if isinstance(tool_call, MagicMock):
                    tool_calls.append({
                        "id": str(tool_call.id),
                        "type": "function",
                        "function": {
                            "name": str(tool_call.function.name),
                            "arguments": str(tool_call.function.arguments)
                        }
                    })
                else:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            return tool_calls if tool_calls else None
        except Exception as e:
            raise ProviderError(f"Failed to extract tool calls: {str(e)}")

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from response."""
        try:
            return TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        except Exception:
            return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """Convert response to dictionary"""
        try:
            return {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [{
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in (choice.message.tool_calls or [])]
                    }
                } for choice in response.choices],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            raise ProviderError(f"Failed to convert response to dict: {str(e)}")

    def _validate_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and format request parameters"""
        validated = {}

        # Handle temperature
        if "temperature" in params and params["temperature"] is not None:
            temp = float(params["temperature"])
            if not 0 <= temp <= 1:
                raise ProviderError("Temperature must be between 0 and 1")
            validated["temperature"] = temp

        # Handle max tokens
        if "max_tokens" in params and params["max_tokens"] is not None:
            max_tokens = int(params["max_tokens"])
            if max_tokens < 1:
                raise ProviderError("max_tokens must be greater than 0")
            validated["max_tokens"] = max_tokens

        # Remove unsupported parameters
        unsupported = ["top_p", "frequency_penalty", "presence_penalty", "top_logprobs"]
        for param in unsupported:
            if param in params:
                params.pop(param)

        return validated

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion"""
        return asyncio.get_event_loop().run_until_complete(
            self._aget_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )

    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion asynchronously"""
        try:
            # Format messages and validate request
            formatted_messages = self._format_messages(messages)
            validated_params = self._validate_request({
                "temperature": temperature,
                "max_tokens": max_tokens
            })

            # Make request to Gemini API
            response = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                **validated_params
            )

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Gemini chat completion failed: {str(e)}")

    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a JSON-formatted chat completion synchronously"""
        return asyncio.get_event_loop().run_until_complete(
            self._aget_json_completion(
                messages=messages,
                model=model,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )

    async def _aget_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a JSON-formatted chat completion asynchronously"""
        try:
            # Add JSON schema instructions
            current_messages = messages.copy()
            schema_json = schema.model_json_schema()
            json_instructions = (
                "Format your response as JSON matching this schema:\n"
                f"{json.dumps(schema_json, indent=2)}\n\n"
                "Respond ONLY with valid JSON matching this schema. No other text."
            )

            # Add or update system message
            system_msg_index = next(
                (i for i, msg in enumerate(current_messages) if msg.role == Role.SYSTEM),
                None
            )
            if system_msg_index is not None:
                current_messages[system_msg_index].content += f"\n\n{json_instructions}"
            else:
                current_messages.insert(0, Message(role=Role.SYSTEM, content=json_instructions))

            # Get completion
            response = await self._aget_chat_completion(
                messages=current_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Validate JSON response
            try:
                json_data = json.loads(response.content)
                schema(**json_data)  # Validate against schema
            except (json.JSONDecodeError, ValidationError) as e:
                raise ProviderError(f"Failed to parse response as valid JSON: {str(e)}")

            return response

        except Exception as e:
            raise ProviderError(f"Gemini JSON completion failed: {str(e)}")

    def _format_tools(self, tools: Sequence[BaseTool]) -> List[Dict[str, Any]]:
        """Format tools for Gemini API"""
        formatted_tools = []
        for tool in tools:
            # Get the schema from the tool's parameters
            if hasattr(tool.parameters, "model_json_schema"):
                schema = tool.parameters.model_json_schema()
            else:
                schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

            # Format the tool spec according to OpenAI's format
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema
                }
            })
        
        return formatted_tools

    def _get_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Get a tool-enabled chat completion synchronously"""
        return asyncio.get_event_loop().run_until_complete(
            self._aget_tool_completion(
                messages=messages,
                model=model,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                format_json=format_json,
                json_schema=json_schema
            )
        )

    async def _aget_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Get a tool-enabled chat completion asynchronously"""
        try:
            current_messages = messages.copy()
            final_tool_calls = []
            max_iterations = 5  # Prevent infinite loops

            # Add tool use instructions
            tool_instructions = (
                "You have access to tools that you can use. When using a tool, format your response as a tool call. "
                "Do not include any text before making tool calls. Make sequential tool calls until you have all "
                "the information needed, then provide your final response."
            )

            # If JSON formatting is requested, add JSON instructions
            if format_json and json_schema:
                schema_json = json_schema.model_json_schema()
                json_instructions = (
                    "\n\nAfter using tools, format your final response as JSON matching this schema:\n"
                    f"{json.dumps(schema_json, indent=2)}\n\n"
                    "Respond ONLY with valid JSON matching this schema. No other text."
                )
                tool_instructions = tool_instructions + json_instructions

            # Add or update system message
            system_msg_index = next(
                (i for i, msg in enumerate(current_messages) if msg.role == Role.SYSTEM),
                None
            )
            if system_msg_index is not None:
                current_messages[system_msg_index].content += f"\n\n{tool_instructions}"
            else:
                current_messages.insert(0, Message(role=Role.SYSTEM, content=tool_instructions))

            iterations = 0
            while iterations < max_iterations:
                iterations += 1
                formatted_messages = self._format_messages(current_messages)
                validated_params = self._validate_request({
                    "temperature": temperature,
                    "max_tokens": max_tokens
                })

                # Format tools for Gemini API
                tool_list = self._format_tools(tools)

                # Make request to Gemini API
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    tools=tool_list,
                    tool_choice="auto",  # Let the model decide when to use tools
                    **validated_params
                )

                content = self._extract_content(response)
                tool_calls = self._extract_tool_calls(response)

                # If no tool calls or we've reached max iterations, we're done
                if not tool_calls or iterations == max_iterations:
                    if format_json and json_schema:
                        try:
                            # Try to parse the content as JSON first
                            json_data = json.loads(content)
                            # Validate against schema
                            schema_instance = json_schema(**json_data)
                            return ModelResponse(
                                content=content,
                                raw_response=self._response_to_dict(response),
                                usage=self._extract_usage(response),
                                tool_calls=final_tool_calls if final_tool_calls else None
                            )
                        except (json.JSONDecodeError, ValidationError):
                            # If content is not valid JSON or doesn't match schema,
                            # make one final attempt to get JSON response
                            json_messages = []
                            
                            # Add system message with JSON instructions
                            schema_json = json_schema.model_json_schema()
                            json_instructions = (
                                "Format your response as JSON matching this schema:\n"
                                f"{json.dumps(schema_json, indent=2)}\n\n"
                                "Respond ONLY with valid JSON matching this schema. No other text."
                            )
                            json_messages.append(Message(role=Role.SYSTEM, content=json_instructions))
                            
                            # Add user message with the request
                            json_messages.append(Message(
                                role=Role.USER,
                                content=f"Create a JSON object with these details: {content}"
                            ))

                            return await self._aget_json_completion(
                                messages=json_messages,
                                model=model,
                                schema=json_schema,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )

                    return ModelResponse(
                        content=content,
                        raw_response=self._response_to_dict(response),
                        usage=self._extract_usage(response),
                        tool_calls=final_tool_calls if final_tool_calls else None
                    )

                # Add assistant's response with tool calls
                current_messages.append(Message(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls
                ))

                # Store tool calls for final response
                final_tool_calls.extend(tool_calls)

                # Process each tool call
                for tool_call in tool_calls:
                    tool = next(
                        (t for t in tools if t.name == tool_call["function"]["name"]),
                        None
                    )

                    if tool:
                        args = json.loads(tool_call["function"]["arguments"])
                        result = await tool.arun(**args)

                        # Add tool result as user message
                        current_messages.append(Message(
                            role=Role.USER,
                            content=f"Tool '{tool_call['function']['name']}' returned: {json.dumps(result) if isinstance(result, dict) else str(result)}"
                        ))

            # If we've reached here, we hit max iterations
            return ModelResponse(
                content="Max iterations reached without completion",
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=final_tool_calls if final_tool_calls else None
            )

        except Exception as e:
            raise ProviderError(f"Gemini tool completion failed: {str(e)}")
