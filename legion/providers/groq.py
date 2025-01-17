"""Groq-specific implementation of the LLM interface"""

import asyncio
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Type

from openai import OpenAI
from pydantic import BaseModel

from ..errors import ProviderError
from ..interface.base import LLMInterface
from ..interface.schemas import Message, ModelResponse, ProviderConfig, Role, TokenUsage
from ..interface.tools import BaseTool
from . import ProviderFactory


class GroqFactory(ProviderFactory):
    """Factory for creating Groq providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new Groq provider instance"""
        # If no config provided, create one with defaults
        if config is None:
            config = ProviderConfig()

        # If no API key in config, try to get from environment
        if not config.api_key:
            config.api_key = os.getenv("GROQ_API_KEY")

        # Set default base URL if not provided
        if not config.base_url:
            config.base_url = "https://api.groq.com/openai/v1"

        return GroqProvider(config=config, **kwargs)


class GroqProvider(LLMInterface):
    """Groq-specific provider implementation"""

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_SYSTEM_INSTRUCTION = (
        "DO NOT attempt to use tools that you do not have access to. "
        "If a user requests something that is outside the scope of your capabilities, "
        "do the best you can with the tools you have available."
    )

    def _setup_client(self) -> None:
        """Initialize Groq client using OpenAI's client"""
        # If no API key in config, try to get from environment
        if not self.config.api_key:
            self.config.api_key = os.getenv("GROQ_API_KEY")

        if not self.config.api_key:
            raise ProviderError("API key is required for Groq provider. Set GROQ_API_KEY environment variable.")

        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or self.DEFAULT_BASE_URL,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize Groq client: {str(e)}")

    async def _asetup_client(self) -> None:
        """Initialize async Groq client"""
        # Groq uses OpenAI's client for both sync and async
        await self._setup_client()

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Groq API"""
        formatted_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Add Groq-specific instruction to system message
                content = f"{msg.content}\n\n{self.GROQ_SYSTEM_INSTRUCTION}" if msg.content else self.GROQ_SYSTEM_INSTRUCTION
                formatted_messages.append({
                    "role": "system",
                    "content": content
                })
                continue

            # Only include required fields for Groq
            message = {
                "role": msg.role.value,
                "content": msg.content or ""
            }

            # Add tool-specific fields only if present
            if msg.role == Role.TOOL and msg.tool_call_id:
                message.update({
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name
                })
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                message["tool_calls"] = msg.tool_calls

            formatted_messages.append(message)

        return formatted_messages

    def _format_tool(self, tool: BaseTool) -> Dict[str, Any]:
        """Format a tool for Groq API"""
        schema = tool.parameters.model_json_schema()
        # Remove title and definitions from schema as they're not needed
        schema.pop("title", None)
        schema.pop("definitions", None)
        
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema
            }
        }

    def _extract_tool_calls(self, response) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from response"""
        if not hasattr(response.choices[0].message, "tool_calls"):
            return None
        if not response.choices[0].message.tool_calls:
            return None
        
        tool_calls = []
        for tool_call in response.choices[0].message.tool_calls:
            call_data = {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            tool_calls.append(call_data)
            
        return tool_calls

    def _extract_content(self, response: Any) -> str:
        """Extract content from Groq response"""
        return response.choices[0].message.content or ""

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from response"""
        usage = response.usage
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return {"content": str(response)}

    def _validate_request(self, **kwargs) -> dict:
        """Validate and modify request parameters for Groq"""
        # Ensure N=1 as Groq doesn't support other values
        if kwargs.get("n", 1) != 1:
            raise ProviderError("Groq only supports n=1")

        # Handle temperature=0 case
        if kwargs.get("temperature", 1.0) == 0:
            kwargs["temperature"] = 1e-8

        # Remove unsupported parameters
        unsupported = ["logprobs", "logit_bias", "top_logprobs"]
        for param in unsupported:
            kwargs.pop(param, None)

        return kwargs

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion"""
        try:
            kwargs = self._validate_request(
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = self.client.chat.completions.create(
                model=model,
                messages=self._format_messages(messages),
                **kwargs
            )

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Groq completion failed: {str(e)}")

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
        """Get completion with tool usage"""
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
        """Get completion with tool usage asynchronously"""
        try:
            current_messages = messages.copy()
            final_tool_calls = []

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

            while True:
                # Format messages for Groq
                formatted_messages = self._format_messages(current_messages)
                kwargs = self._validate_request(
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Format tools for Groq
                formatted_tools = [self._format_tool(tool) for tool in tools]

                # Create request
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",  # Use the specified model
                    messages=formatted_messages,
                    tools=formatted_tools,
                    tool_choice="auto",
                    **kwargs
                )

                content = self._extract_content(response)
                tool_calls = self._extract_tool_calls(response)

                # If no tool calls, we're done
                if not tool_calls:
                    # If JSON formatting requested, format the final response
                    if format_json and json_schema:
                        json_messages = self._create_json_conversation(current_messages, json_schema)
                        json_response = await self._aget_json_completion(
                            messages=json_messages,
                            model=model,
                            schema=json_schema,
                            temperature=0.0,
                            max_tokens=max_tokens
                        )
                        # Preserve tool calls in the JSON response
                        return ModelResponse(
                            content=json_response.content,
                            raw_response=json_response.raw_response,
                            usage=json_response.usage,
                            tool_calls=final_tool_calls
                        )

                    return ModelResponse(
                        content=content,
                        raw_response=self._response_to_dict(response),
                        tool_calls=final_tool_calls if final_tool_calls else None,
                        usage=self._extract_usage(response)
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
                        result = await tool.arun(**args)  # Use async tool call

                        # Add tool response to conversation
                        current_messages.append(Message(
                            role=Role.TOOL,  # Use Role.TOOL for OpenAI compatibility
                            content=str(result),
                            tool_call_id=tool_call["id"],
                            name=tool_call["function"]["name"]
                        ))

        except Exception as e:
            raise ProviderError(f"Groq tool completion failed: {str(e)}")

    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON"""
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
        """Get a chat completion formatted as JSON asynchronously"""
        try:
            # Add JSON formatting instructions
            json_instructions = (
                "Format your response as JSON matching this schema:\n"
                f"{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
                "Respond ONLY with valid JSON matching this schema. No other text."
            )

            # Add or update system message
            current_messages = messages.copy()
            system_msg_index = next(
                (i for i, msg in enumerate(current_messages) if msg.role == Role.SYSTEM),
                None
            )
            if system_msg_index is not None:
                current_messages[system_msg_index].content += f"\n\n{json_instructions}"
            else:
                current_messages.insert(0, Message(role=Role.SYSTEM, content=json_instructions))

            # Format messages for Groq
            formatted_messages = self._format_messages(current_messages)
            kwargs = self._validate_request(
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Create request
            response = self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                response_format={"type": "json_object"},
                **kwargs
            )

            # Extract and validate content
            content = self._extract_content(response)
            try:
                data = json.loads(content)
                schema.model_validate(data)
            except Exception as e:
                raise ProviderError(f"Invalid JSON response: {str(e)}")

            return ModelResponse(
                content=content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )

        except Exception as e:
            raise ProviderError(f"Groq JSON completion failed: {str(e)}")

    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion asynchronously"""
        # For now, just use sync version since Groq uses OpenAI's client
        return self._get_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
