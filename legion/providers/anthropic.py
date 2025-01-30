import json
from typing import Any, Dict, List, Optional, Sequence, Type

import anthropic
from pydantic import BaseModel

from ..errors import ProviderError
from ..interface.base import LLMInterface
from ..interface.schemas import (
    ChatParameters,
    Message,
    ModelResponse,
    ProviderConfig,
    Role,
    TokenUsage,
)
from ..interface.tools import BaseTool
from .factory import ProviderFactory


class AnthropicFactory(ProviderFactory):
    """Factory for creating Anthropic providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new Anthropic provider instance"""
        return AnthropicProvider(config=config, **kwargs)

class AnthropicProvider(LLMInterface):
    """Anthropic-specific implementation of the LLM interface"""

    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, config: Optional[ProviderConfig] = None, **kwargs):
        if not config or not config.api_key:
            raise ProviderError("API key is required for Anthropic")
        super().__init__(config=config, **kwargs)

    def _setup_client(self) -> None:
        """Initialize Anthropic client"""
        try:
            self.client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize Anthropic client: {str(e)}")

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Anthropic format"""
        anthropic_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue  # System messages handled separately

            # Initialize message
            formatted_msg = {"role": "user" if msg.role in [Role.USER, Role.TOOL] else "assistant"}

            # Handle different message types
            if msg.tool_calls:
                formatted_msg["content"] = [{
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"])
                } for tool_call in msg.tool_calls]
            elif msg.role == Role.TOOL and msg.tool_call_id:
                formatted_msg["content"] = [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content
                }]
            else:
                # For basic messages, use string content
                formatted_msg["content"] = msg.content

            anthropic_messages.append(formatted_msg)

        return anthropic_messages

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        params: ChatParameters
    ) -> ModelResponse:
        """Get a basic chat completion"""
        try:
            # Extract system message if present
            system_message = next(
                (msg.content for msg in messages if msg.role == Role.SYSTEM),
                None
            )

            # Debug: Print formatted messages
            if self.debug:
                print("\nSending to Anthropic API:")
                print(f"Model: {model}")
                print(f"Messages: {self._format_messages(messages)}")
                print(f"System: {system_message}")
                print(f"Temperature: {params.temperature}")
                print(f"Max Tokens: {params.max_tokens or self.DEFAULT_MAX_TOKENS}")

            # Create request parameters
            request_params = {
                "model": model,
                "messages": self._format_messages(messages),
                "temperature": params.temperature,
                "max_tokens": params.max_tokens or self.DEFAULT_MAX_TOKENS
            }

            # Only add system if it's not None
            if system_message is not None:
                request_params["system"] = system_message

            response = self.client.messages.create(**request_params)

            # Debug: Print raw response
            if self.debug:
                print("\nReceived from Anthropic API:")
                print(f"Response Type: {type(response)}")
                print(f"Response Content: {response.content}")
                print(f"Response Model Dump: {response.model_dump()}")

            model_response = ModelResponse(
                content=self._extract_content(response),
                raw_response=response.model_dump(),
                usage=self._extract_usage(response),
                tool_calls=None
            )

            # Debug: Print final model response
            if self.debug:
                print("\nFinal ModelResponse:")
                print(f"Content: {model_response.content}")
                print(f"Raw Response: {model_response.raw_response}")
                print(f"Usage: {model_response.usage}")

            return model_response
        except Exception as e:
            raise ProviderError(f"Anthropic completion failed: {str(e)}")

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
        """Get a chat completion with tool use"""
        try:
            # Extract system message if present
            system_message = next(
                (msg.content for msg in messages if msg.role == Role.SYSTEM),
                None
            )

            # Add tool use instructions to system message
            tool_instructions = (
                "If you need to use tools, do not include any text before making tool calls. "
                "Simply make sequential tool calls until you have all the information needed, "
                "then provide your final response."
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

            if system_message:
                system_message = f"{system_message}\n\n{tool_instructions}"
            else:
                system_message = tool_instructions

            # Format tools for Anthropic
            anthropic_tools = []
            for tool in tools:
                schema = tool.parameters.model_json_schema()
                anthropic_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", [])
                    }
                })

            # Initialize conversation history
            conversation = []
            current_messages = messages.copy()
            final_response = None
            final_tool_calls = []

            while True:
                # Format current messages
                formatted_messages = []
                current_interaction = []

                for msg in current_messages:
                    if msg.role == Role.SYSTEM:
                        continue

                    if msg.role == Role.USER and current_interaction:
                        formatted_messages.extend(self._format_messages(current_interaction))
                        current_interaction = []

                    current_interaction.append(msg)

                if current_interaction:
                    formatted_messages.extend(self._format_messages(current_interaction))

                # Create request
                request_kwargs = {
                    "model": model,
                    "messages": formatted_messages,
                    "tools": anthropic_tools,
                    "temperature": temperature,
                    "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
                    "system": system_message,
                    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}
                }

                if self.debug:
                    print("\nSending messages to Anthropic:")
                    for msg in formatted_messages:
                        print(f"Role: {msg['role']}")
                        print(f"Content: {msg['content']}\n")

                response = self.client.messages.create(**request_kwargs)

                content = self._extract_content(response)
                tool_calls = self._extract_tool_calls(response)

                # Add assistant's response to conversation
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls
                )
                conversation.append(assistant_msg)
                current_messages.append(assistant_msg)

                if tool_calls:
                    final_tool_calls.extend(tool_calls)
                    # Process tool calls
                    for tool_call in tool_calls:
                        tool = next(
                            (t for t in tools if t.name == tool_call["function"]["name"]),
                            None
                        )
                        if tool:
                            try:
                                args = json.loads(tool_call["function"]["arguments"])
                                result = tool(**args)

                                # Add tool response to conversation
                                tool_msg = Message(
                                    role=Role.TOOL,
                                    content=str(result),
                                    tool_call_id=tool_call["id"],
                                    name=tool_call["function"]["name"]
                                )
                                conversation.append(tool_msg)
                                current_messages.append(tool_msg)
                            except Exception as e:
                                raise ProviderError(f"Error executing {tool.name}: {str(e)}")
                else:
                    # No more tool calls, this is our final response
                    final_response = response
                    break

            if self.debug and final_tool_calls:
                print("\nTool calls triggered:")
                for call in final_tool_calls:
                    print(f"- {call['function']['name']}: {call['function']['arguments']}")
                if content:
                    print(f"\nAssistant message: {content}")

            # If JSON formatting was requested, validate the response
            content = self._extract_content(final_response)
            if format_json and json_schema:
                try:
                    data = json.loads(content)
                    json_schema.model_validate(data)
                except Exception as e:
                    raise ProviderError(f"Invalid JSON response: {str(e)}")

            return ModelResponse(
                content=content,
                raw_response=final_response.model_dump(),
                usage=self._extract_usage(final_response),
                tool_calls=final_tool_calls if final_tool_calls else None
            )
        except Exception as e:
            raise ProviderError(f"Anthropic tool completion failed: {str(e)}")

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

            # Create system message with JSON instructions
            system_message = formatting_prompt

            response = self.client.messages.create(
                model=model,
                messages=self._format_messages(messages),
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens or self.DEFAULT_MAX_TOKENS
            )

            # Validate response against schema
            content = self._extract_content(response)
            try:
                data = json.loads(content)
                schema.model_validate(data)
            except Exception as e:
                raise ProviderError(f"Invalid JSON response: {str(e)}")

            return ModelResponse(
                content=content,
                raw_response=response.model_dump(),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Anthropic JSON completion failed: {str(e)}")

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from Anthropic response"""
        tool_calls = []
        tool_call_count = 0

        for block in response.content:
            if block.type == "tool_use":
                # Generate a unique ID if none provided
                call_id = getattr(block, "id", f"call_{tool_call_count}")
                tool_call_count += 1

                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })

                if self.debug:
                    print(f"\nExtracted tool call: {json.dumps(tool_calls[-1], indent=2)}")

        return tool_calls if tool_calls else None

    def _extract_content(self, response: Any) -> str:
        """Extract content from Anthropic response"""
        if not hasattr(response, "content"):
            return ""

        if isinstance(response.content, list):
            text_blocks = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    text_blocks.append(block.text)
                elif self.debug:
                    print(f"Skipping non-text block of type: {getattr(block, 'type', 'unknown')}")
            return " ".join(text_blocks).strip()

        return str(response.content).strip()

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from Anthropic response"""
        return TokenUsage(
            prompt_tokens=getattr(response.usage, "input_tokens", 0),
            completion_tokens=getattr(response.usage, "output_tokens", 0),
            total_tokens=(
                getattr(response.usage, "input_tokens", 0) +
                getattr(response.usage, "output_tokens", 0)
            )
        )

    async def _asetup_client(self) -> None:
        """Initialize Anthropic client asynchronously"""
        await self._setup_client()

    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion asynchronously"""
        params = ChatParameters(
            temperature=temperature,
            max_tokens=max_tokens
        )
        return self._get_chat_completion(messages, model, params)

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
        """Get a chat completion with tool use asynchronously"""
        return self._get_tool_completion(
            messages=messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            format_json=format_json,
            json_schema=json_schema
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
        return self._get_json_completion(messages, model, schema, temperature, max_tokens)
