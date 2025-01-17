"""Tests for the Anthropic provider implementation"""

import json
import os
from typing import List, Optional

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from legion.errors import ProviderError
from legion.interface.schemas import ChatParameters, Message, ModelResponse, ProviderConfig, Role
from legion.interface.tools import BaseTool
from legion.providers.anthropic import AnthropicFactory, AnthropicProvider

# Load environment variables
load_dotenv()

class MockToolParams(BaseModel):
    """Parameters for mock tool"""

    input: str = Field(description="Input to process")


class MockTool(BaseTool):
    """Mock tool for testing"""

    def __init__(self, name="mock_tool", description="A mock tool"):
        super().__init__(
            name=name,
            description=description,
            parameters=MockToolParams
        )

    def __call__(self, **kwargs):
        return self.run(**kwargs)

    def run(self, **kwargs):
        return "Mock tool response"


class TestResponse(BaseModel):
    """Test response schema"""

    message: str = Field(description="A test message")
    score: float = Field(description="A test score between 0 and 1", ge=0, le=1)
    tags: Optional[List[str]] = Field(description="Optional list of tags")


@pytest.fixture
def provider():
    """Create a test provider instance"""
    config = ProviderConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com",
        timeout=60,
        max_retries=3
    )
    return AnthropicProvider(config=config, debug=True)


@pytest.fixture
def factory():
    """Create a test factory instance"""
    return AnthropicFactory()


def test_provider_creation(factory):
    """Test provider creation through factory"""
    config = ProviderConfig(api_key="test_key")
    provider = factory.create_provider(config)
    assert isinstance(provider, AnthropicProvider)


def test_provider_initialization(provider):
    """Test provider initialization"""
    assert isinstance(provider, AnthropicProvider)
    assert provider.client is not None


def test_message_formatting(provider):
    """Test message formatting"""
    messages = [
        Message(role=Role.SYSTEM, content="System message"),
        Message(role=Role.USER, content="User message"),
        Message(role=Role.ASSISTANT, content="Assistant message"),
        Message(
            role=Role.USER,
            content="Tool message",
            tool_calls=[{
                "id": "call_1",
                "function": {
                    "name": "test_tool",
                    "arguments": json.dumps({"input": "test"})
                }
            }]
        ),
        Message(
            role=Role.TOOL,
            content="Tool result",
            tool_call_id="call_1",
            name="test_tool"
        )
    ]
    formatted = provider._format_messages(messages)

    # System message should not be in formatted messages
    assert len(formatted) == 4
    assert formatted[0]["role"] == "user"
    assert formatted[1]["role"] == "assistant"
    assert formatted[2]["role"] == "user"
    assert formatted[3]["role"] == "user"  # Tool messages are treated as user messages

    # Basic messages use string content
    assert formatted[0]["content"] == "User message"
    assert formatted[1]["content"] == "Assistant message"

    # Tool messages use structured content
    assert isinstance(formatted[2]["content"], list)
    assert formatted[2]["content"][0]["type"] == "tool_use"
    assert formatted[2]["content"][0]["id"] == "call_1"
    assert formatted[2]["content"][0]["name"] == "test_tool"
    assert formatted[2]["content"][0]["input"] == {"input": "test"}

    assert isinstance(formatted[3]["content"], list)
    assert formatted[3]["content"][0]["type"] == "tool_result"
    assert formatted[3]["content"][0]["tool_use_id"] == "call_1"
    assert formatted[3]["content"][0]["content"] == "Tool result"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
def test_basic_completion(provider):
    """Test basic text completion"""
    messages = [
        Message(role=Role.USER, content="Hello, Claude")
    ]
    params = ChatParameters(
        temperature=0,
        max_tokens=1024
    )
    response = provider._get_chat_completion(
        messages=messages,
        model="claude-3-haiku-20240307",
        params=params
    )

    # Test response structure
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.tool_calls is None

    # Test raw response
    assert isinstance(response.raw_response, dict)
    assert "content" in response.raw_response
    assert isinstance(response.raw_response["content"], list)
    assert response.raw_response["content"][0]["type"] == "text"

    # Test usage
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
@pytest.mark.asyncio
async def test_tool_completion(provider):
    """Test completion with tool use"""
    tool = MockTool()
    messages = [
        Message(role=Role.USER, content="Use the mock tool with input='test'")
    ]
    response = await provider.complete(
        messages=messages,
        model="claude-3-haiku-20240307",
        tools=[tool],
        temperature=0.7
    )
    assert isinstance(response, ModelResponse)
    assert response.content
    assert response.usage
    assert response.usage.total_tokens > 0
    assert response.raw_response
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "mock_tool"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
@pytest.mark.asyncio
async def test_json_completion(provider):
    """Test JSON completion"""
    messages = [
        Message(
            role=Role.USER,
            content="Generate a test response with message='Hello', score=0.9, tags=['test']"
        )
    ]
    response = await provider.complete(
        messages=messages,
        model="claude-3-haiku-20240307",
        temperature=0.1,
        response_schema=TestResponse
    )
    assert response.content
    assert response.raw_response

    # Verify JSON is valid and matches schema
    data = json.loads(response.content)
    test_response = TestResponse(**data)
    assert test_response.message
    assert 0 <= test_response.score <= 1
    assert isinstance(test_response.tags, list) or test_response.tags is None


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
@pytest.mark.asyncio
async def test_tool_and_json_completion(provider):
    """Test combining tool use with JSON response formatting"""
    tool = MockTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools and returns structured data."),
        Message(role=Role.USER, content="Use the mock tool with input='test', then format the response as a test response")
    ]
    response = await provider.complete(
        messages=messages,
        model="claude-3-haiku-20240307",
        tools=[tool],
        temperature=0.7,
        response_schema=TestResponse
    )
    assert response.content
    assert response.usage
    assert response.usage.total_tokens > 0
    assert response.raw_response
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0

    # Verify JSON response
    data = json.loads(response.content)
    test_response = TestResponse(**data)
    assert test_response.message
    assert 0 <= test_response.score <= 1
    assert isinstance(test_response.tags, list) or test_response.tags is None


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
@pytest.mark.asyncio
async def test_async_completion(provider):
    """Test async completion functionality"""
    messages = [
        Message(role=Role.USER, content="Hello, Claude")
    ]
    response = await provider.acomplete(
        messages=messages,
        model="claude-3-haiku-20240307",
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.tool_calls is None
    assert response.usage is not None
    assert response.usage.total_tokens > 0


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
@pytest.mark.asyncio
async def test_system_message_handling(provider):
    """Test system message handling"""
    messages = [
        Message(role=Role.SYSTEM, content="You are Claude, a helpful AI assistant created by Anthropic."),
        Message(role=Role.USER, content="Who are you?")
    ]
    response = await provider.complete(
        messages=messages,
        model="claude-3-haiku-20240307",
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    # Check that the response mentions being Claude or an Anthropic assistant
    assert any(word in response.content.lower() for word in ["claude", "anthropic", "assist"])


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
@pytest.mark.asyncio
async def test_invalid_model(provider):
    """Test error handling for invalid model"""
    messages = [Message(role=Role.USER, content="Test")]
    with pytest.raises(ProviderError):
        await provider.complete(
            messages=messages,
            model="invalid-model",
            temperature=0.7
        )


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
@pytest.mark.asyncio
async def test_invalid_api_key():
    """Test error handling for invalid API key"""
    config = ProviderConfig(api_key="invalid_key")
    provider = AnthropicProvider(config=config)
    messages = [Message(role=Role.USER, content="Test")]
    with pytest.raises(ProviderError):
        await provider.complete(
            messages=messages,
            model="claude-3-haiku-20240307",
            temperature=0.7
        )


if __name__ == "__main__":
    pytest.main([__file__])
