"""Groq-specific implementation of the LLM interface"""

import asyncio
import json
import os
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

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from Groq response"""
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
        """Convert OpenAI response to dictionary"""
        return {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                            for tool_call in (choice.message.tool_calls or [])
                        ] if choice.message.tool_calls else None
                    }
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

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
            # Initialize conversation
            current_messages = messages.copy()

            while True:
                if self.debug:
                    print(f"\nSending request to Groq with {len(current_messages)} messages...")

                kwargs = self._validate_request(
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Get response with tools
                response = self.client.chat.completions.create(
                    model=model,
                    messages=self._format_messages(current_messages),
                    tools=[tool.get_schema() for tool in tools if tool.parameters],
                    tool_choice="auto",
                    **kwargs
                )

                # Get assistant's message and tool calls
                assistant_message = response.choices[0].message
                content = assistant_message.content or ""  # Ensure content is never None
                tool_calls = self._extract_tool_calls(response)

                if self.debug:
                    if tool_calls:
                        print("\nGroq tool calls triggered:")
                        for call in tool_calls:
                            print(f"- {call['function']['name']}: {call['function']['arguments']}")
                    else:
                        print("\nNo tool calls in Groq response")

                # Add assistant's response to conversation
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls
                )
                current_messages.append(assistant_msg)

                # If no tool calls, this is our final response
                if not tool_calls:
                    if self.debug:
                        print("\nFinal response received from Groq")
                    return ModelResponse(
                        content=content,
                        raw_response=self._response_to_dict(response),
                        usage=self._extract_usage(response),
                        tool_calls=None
                    )

                # Process tool calls
                for tool_call in tool_calls:
                    tool = next(
                        (t for t in tools if t.name == tool_call["function"]["name"]),
                        None
                    )
                    if tool:
                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                            result = await tool(**args) if asyncio.iscoroutinefunction(tool) else tool(**args)

                            if self.debug:
                                print(f"\nTool {tool.name} returned: {result}")

                            # Add tool response to conversation
                            tool_msg = Message(
                                role=Role.TOOL,
                                content=str(result),
                                tool_call_id=tool_call["id"],
                                name=tool_call["function"]["name"]
                            )
                            current_messages.append(tool_msg)
                        except Exception as e:
                            raise ProviderError(f"Error executing {tool.name}: {str(e)}")

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
        try:
            # Get generic JSON formatting prompt
            formatting_prompt = self._get_json_formatting_prompt(schema, messages[-1].content)

            # Create messages for Groq
            groq_messages = [
                {"role": "system", "content": formatting_prompt}
            ]

            # Add remaining messages, skipping system and only including required fields
            groq_messages.extend([
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
                if msg.role != Role.SYSTEM
            ])

            kwargs = self._validate_request(
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = self.client.chat.completions.create(
                model=model,
                messages=groq_messages,
                response_format={"type": "json_object"},
                **kwargs
            )

            # Validate response against schema
            content = response.choices[0].message.content
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

    async def _aget_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON asynchronously"""
        # For now, just use sync version since Groq uses OpenAI's client
        return self._get_json_completion(
            messages=messages,
            model=model,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens
        )
