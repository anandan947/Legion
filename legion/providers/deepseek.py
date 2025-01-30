import json
from typing import Any, Dict, List, Optional, Sequence, Type

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from ..errors import ProviderError
from ..interface.base import LLMInterface
from ..interface.schemas import (
    Message,
    ModelResponse,
    ProviderConfig,
    Role,
    TokenUsage,
)
from ..interface.tools import BaseTool
from .factory import ProviderFactory


class DeepSeekFactory(ProviderFactory):
    """Factory for creating DeepSeek providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new DeepSeek provider instance"""
        return DeepSeekProvider(config=config or ProviderConfig(), **kwargs)


class DeepSeekProvider(LLMInterface):
    """DeepSeek-specific implementation of the LLM interface"""

    def __init__(self, config: ProviderConfig, debug: bool = False):
        """Initialize provider with both sync and async clients"""
        if not config.base_url:
            config.base_url = "https://api.deepseek.com"
        super().__init__(config, debug)
        self._async_client = None  # Initialize async client lazily

    def _setup_client(self) -> None:
        """Initialize DeepSeek client"""
        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize DeepSeek client: {str(e)}")

    async def _asetup_client(self) -> None:
        """Initialize async DeepSeek client"""
        try:
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize async DeepSeek client: {str(e)}")

    async def _ensure_async_client(self) -> None:
        """Ensure async client is initialized"""
        if self._async_client is None:
            await self._asetup_client()

    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion asynchronously"""
        try:
            await self._ensure_async_client()
            response = await self._async_client.chat.completions.create(
                model=model or "deepseek-chat",
                messages=[msg.model_dump() for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek async completion failed: {str(e)}")

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
            await self._ensure_async_client()
            tools_dict = [tool.to_dict() for tool in tools]
            response = await self._async_client.chat.completions.create(
                model=model or "deepseek-chat",
                messages=[msg.model_dump() for msg in messages],
                tools=tools_dict,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if format_json else None
            )

            tool_calls = self._extract_tool_calls(response)
            content = self._extract_content(response)

            return ModelResponse(
                content=content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=tool_calls
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek async tool completion failed: {str(e)}")

    async def _aget_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None,
        preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Get a JSON-formatted chat completion asynchronously"""
        try:
            await self._ensure_async_client()
            response = await self._async_client.chat.completions.create(
                model=model or "deepseek-chat",
                messages=[msg.model_dump() for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=preserve_tool_calls
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek async JSON completion failed: {str(e)}")

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the API request"""
        return [msg.model_dump() for msg in messages]

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion synchronously"""
        try:
            response = self.client.chat.completions.create(
                model=model or "deepseek-chat",
                messages=[msg.model_dump() for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek completion failed: {str(e)}")

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
        try:
            tools_dict = [tool.to_dict() for tool in tools]
            response = self.client.chat.completions.create(
                model=model or "deepseek-chat",
                messages=[msg.model_dump() for msg in messages],
                tools=tools_dict,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if format_json else None
            )

            tool_calls = self._extract_tool_calls(response)
            content = self._extract_content(response)

            return ModelResponse(
                content=content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=tool_calls
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek tool completion failed: {str(e)}")

    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None,
        preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Get a JSON-formatted chat completion synchronously"""
        try:
            response = self.client.chat.completions.create(
                model=model or "deepseek-chat",
                messages=[msg.model_dump() for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=preserve_tool_calls
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek JSON completion failed: {str(e)}")

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from the response"""
        if not hasattr(response.choices[0].message, "tool_calls") or not response.choices[0].message.tool_calls:
            return None
        return [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            for tool_call in response.choices[0].message.tool_calls
        ]

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response"""
        return response.choices[0].message.content or ""

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from the response"""
        usage = response.usage
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """Convert response to a dictionary"""
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
                        **({"tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                            for tool_call in choice.message.tool_calls
                        ]} if hasattr(choice.message, "tool_calls") and choice.message.tool_calls else {})
                    },
                    "finish_reason": choice.finish_reason
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        } 