"""Tests for the Groq provider implementation"""

import json
import os
from typing import List, Optional

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from legion.errors import ProviderError
from legion.interface.schemas import Message, ProviderConfig, Role
from legion.interface.tools import BaseTool
from legion.providers.groq import GroqFactory, GroqProvider

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

    async def arun(self, **kwargs):
        return self.run(**kwargs)


class TestResponse(BaseModel):
    """Test response schema"""

    message: str = Field(description="A test message")
    score: float = Field(description="A test score between 0 and 1", ge=0, le=1)
    tags: Optional[List[str]] = Field(description="Optional list of tags")


@pytest.fixture
def provider():
    """Create a test provider instance"""
    config = ProviderConfig(
        api_key=os.environ.get("GROQ_API_KEY", "test_key"),
        base_url="https://api.groq.com/openai/v1"
    )
    return GroqProvider(config=config, debug=True)


@pytest.fixture
def factory():
    """Create a test factory instance"""
    return GroqFactory()


def test_provider_creation(factory):
    """Test provider creation through factory"""
    config = ProviderConfig(api_key="test_key")
    provider = factory.create_provider(config)
    assert isinstance(provider, GroqProvider)


def test_provider_initialization():
    """Test provider initialization with invalid config"""
    # Clear GROQ_API_KEY from env temporarily
    key = os.environ.pop("GROQ_API_KEY", None)
    try:
        with pytest.raises(ProviderError):
            provider = GroqProvider(ProviderConfig(api_key=""))
            provider._setup_client()  # Force client setup to test API key validation
    finally:
        if key:
            os.environ["GROQ_API_KEY"] = key


def test_message_formatting(provider):
    """Test message formatting"""
    messages = [
        Message(role=Role.SYSTEM, content="System message"),
        Message(role=Role.USER, content="User message"),
        Message(role=Role.ASSISTANT, content="Assistant message")
    ]
    formatted = provider._format_messages(messages)

    assert len(formatted) == 3
    assert formatted[0]["role"] == "system"
    assert formatted[1]["role"] == "user"
    assert formatted[2]["role"] == "assistant"
    assert provider.GROQ_SYSTEM_INSTRUCTION in formatted[0]["content"]

    # Verify only required fields are included
    for msg in formatted[1:]:  # Skip system message
        assert set(msg.keys()).issubset({"role", "content", "tool_calls", "tool_call_id", "name"})


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No Groq API key available")
def test_basic_completion(provider):
    """Test basic text completion"""
    messages = [
        Message(role=Role.USER, content="Say hello!")
    ]
    response = provider._get_chat_completion(
        messages=messages,
        model="mixtral-8x7b-32768",
        temperature=0.7
    )
    assert response.content
    assert response.usage
    assert response.usage.total_tokens > 0
    assert isinstance(response.raw_response, dict)


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No Groq API key available")
@pytest.mark.asyncio
async def test_tool_completion(provider):
    """Test completion with tool use"""
    messages = [
        Message(role=Role.USER, content="Use the mock tool with input='test'")
    ]
    tool = MockTool()
    response = await provider._aget_tool_completion(
        messages=messages,
        model="llama-3.1-8b-instant",
        tools=[tool],
        temperature=0.7
    )
    assert response.content
    assert response.usage
    assert response.usage.total_tokens > 0
    assert isinstance(response.raw_response, dict)


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No Groq API key available")
def test_json_completion(provider):
    """Test JSON completion"""
    messages = [
        Message(
            role=Role.USER,
            content="Generate a test response with message='Hello', score=0.9, tags=['test']"
        )
    ]
    response = provider._get_json_completion(
        messages=messages,
        model="llama-3.1-8b-instant",
        schema=TestResponse,
        temperature=0.7
    )
    assert response.content
    assert isinstance(response.raw_response, dict)

    # Verify JSON is valid and matches schema
    data = json.loads(response.content)
    test_response = TestResponse(**data)
    assert test_response.message
    assert 0 <= test_response.score <= 1
    assert isinstance(test_response.tags, list) or test_response.tags is None


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No Groq API key available")
def test_temperature_validation(provider):
    """Test temperature validation and adjustment"""
    messages = [Message(role=Role.USER, content="Test")]

    # Test temperature=0 gets converted to 1e-8
    response = provider._get_chat_completion(
        messages=messages,
        model="mixtral-8x7b-32768",
        temperature=0
    )
    assert response.content
    assert isinstance(response.raw_response, dict)


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No Groq API key available")
def test_error_handling(provider):
    """Test error handling for invalid requests"""
    messages = [Message(role=Role.USER, content="Test")]

    # Test invalid model
    with pytest.raises(ProviderError):
        provider._get_chat_completion(
            messages=messages,
            model="invalid-model",
            temperature=0.7
        )

    # Test invalid temperature
    with pytest.raises(ProviderError):
        provider._get_chat_completion(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=2.5
        )


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="No Groq API key available")
@pytest.mark.asyncio
async def test_tool_and_json_completion(provider):
    """Test completion with both tool use and JSON formatting"""
    messages = [
        Message(
            role=Role.USER,
            content="Use the mock tool with input='test' and format the result as a test response"
        )
    ]
    tool = MockTool()
    response = await provider._aget_tool_completion(
        messages=messages,
        model="llama-3.1-8b-instant",
        tools=[tool],
        temperature=0.7,
        format_json=True,
        json_schema=TestResponse
    )
    
    assert response.content
    assert response.usage
    assert response.usage.total_tokens > 0
    assert isinstance(response.raw_response, dict)
    
    # Verify tool calls were made
    assert response.tool_calls
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "mock_tool"
    
    # Verify JSON is valid and matches schema
    data = json.loads(response.content)
    test_response = TestResponse(**data)
    assert test_response.message
    assert 0 <= test_response.score <= 1
    assert isinstance(test_response.tags, list) or test_response.tags is None


if __name__ == "__main__":
    pytest.main([__file__])
