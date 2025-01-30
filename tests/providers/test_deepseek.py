import os
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from legion.errors import ProviderError
from legion.interface.schemas import Message, ModelResponse, ProviderConfig, Role, TokenUsage
from legion.interface.tools import BaseTool
from legion.providers.deepseek import DeepSeekFactory, DeepSeekProvider

# Load environment variables
load_dotenv()

class TestSchema(BaseModel):
    name: str
    age: int
    hobbies: List[str]

class SimpleToolParams(BaseModel):
    message: str

class SimpleTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="simple_tool",
            description="A simple test tool",
            parameters=SimpleToolParams
        )

    def run(self, message: str) -> str:
        """Implement the sync run method"""
        return f"Tool response: {message}"

    async def arun(self, message: str) -> str:
        """Implement the async run method"""
        return self.run(message)

@pytest.fixture
def provider():
    config = ProviderConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", "test-key"),
        base_url="https://api.deepseek.com"
    )
    return DeepSeekProvider(config=config)

@pytest.fixture
def factory():
    return DeepSeekFactory()

def test_provider_initialization(provider):
    assert isinstance(provider, DeepSeekProvider)
    assert provider.client is not None

def test_factory_creation(factory):
    config = ProviderConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", "test-key"),
        base_url="https://api.deepseek.com"
    )
    provider = factory.create_provider(config=config)
    assert isinstance(provider, DeepSeekProvider)

@pytest.mark.asyncio
async def test_basic_completion(provider):
    messages = [
        Message(role=Role.USER, content="Say 'Hello, World!'")
    ]
    
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content="Hello, World!"))]
    mock_response.usage = AsyncMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_response.id = "test-id"
    mock_response.object = "chat.completion"
    mock_response.created = 1234567890
    mock_response.model = "deepseek-chat"

    with patch.object(provider._async_client.chat.completions, 'create', return_value=mock_response):
        response = await provider.acomplete(
            messages=messages,
            model="deepseek-chat",
            temperature=0
        )
        
        assert isinstance(response, ModelResponse)
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.usage is not None
        assert response.tool_calls is None

@pytest.mark.asyncio
async def test_tool_completion(provider):
    tool = SimpleTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools when appropriate."),
        Message(role=Role.USER, content="Use the simple tool to say hello")
    ]

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(
        message=AsyncMock(
            content="",
            tool_calls=[
                AsyncMock(
                    id="call_1",
                    type="function",
                    function=AsyncMock(
                        name="simple_tool",
                        arguments='{"message": "hello"}'
                    )
                )
            ]
        )
    )]
    mock_response.usage = AsyncMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_response.id = "test-id"
    mock_response.object = "chat.completion"
    mock_response.created = 1234567890
    mock_response.model = "deepseek-chat"

    with patch.object(provider._async_client.chat.completions, 'create', return_value=mock_response):
        response = await provider.acomplete(
            messages=messages,
            model="deepseek-chat",
            tools=[tool],
            temperature=0
        )
        
        assert isinstance(response, ModelResponse)
        assert response.tool_calls is not None
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0]["function"]["name"] == "simple_tool"

@pytest.mark.asyncio
async def test_json_completion(provider):
    messages = [
        Message(
            role=Role.USER,
            content="Give me information about a person named John who is 25 and likes reading and gaming"
        )
    ]

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(
        message=AsyncMock(
            content='{"name": "John", "age": 25, "hobbies": ["reading", "gaming"]}'
        )
    )]
    mock_response.usage = AsyncMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_response.id = "test-id"
    mock_response.object = "chat.completion"
    mock_response.created = 1234567890
    mock_response.model = "deepseek-chat"

    with patch.object(provider._async_client.chat.completions, 'create', return_value=mock_response):
        response = await provider.acomplete(
            messages=messages,
            model="deepseek-chat",
            response_schema=TestSchema,
            temperature=0
        )
        
        assert isinstance(response, ModelResponse)
        assert isinstance(response.content, str)
        data = TestSchema.model_validate_json(response.content)
        assert data.name == "John"
        assert data.age == 25
        assert "reading" in data.hobbies
        assert "gaming" in data.hobbies

@pytest.mark.asyncio
async def test_tool_and_json_completion(provider):
    tool = SimpleTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools when appropriate."),
        Message(role=Role.USER, content="Use the simple tool to say hello and format the response as JSON")
    ]

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(
        message=AsyncMock(
            content='{"name": "Test", "age": 30, "hobbies": ["coding"]}',
            tool_calls=[
                AsyncMock(
                    id="call_1",
                    type="function",
                    function=AsyncMock(
                        name="simple_tool",
                        arguments='{"message": "hello"}'
                    )
                )
            ]
        )
    )]
    mock_response.usage = AsyncMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_response.id = "test-id"
    mock_response.object = "chat.completion"
    mock_response.created = 1234567890
    mock_response.model = "deepseek-chat"

    with patch.object(provider._async_client.chat.completions, 'create', return_value=mock_response):
        response = await provider.acomplete(
            messages=messages,
            model="deepseek-chat",
            tools=[tool],
            response_schema=TestSchema,
            temperature=0
        )
        
        assert isinstance(response, ModelResponse)
        assert response.tool_calls is not None
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0]["function"]["name"] == "simple_tool"
        data = TestSchema.model_validate_json(response.content)
        assert isinstance(data.name, str)
        assert isinstance(data.age, int)
        assert isinstance(data.hobbies, list)

@pytest.mark.asyncio
async def test_invalid_api_key():
    config = ProviderConfig(
        api_key="invalid_key",
        base_url="https://api.deepseek.com"
    )
    provider = DeepSeekProvider(config=config)
    messages = [Message(role=Role.USER, content="Hello")]
    
    with pytest.raises(ProviderError):
        await provider.acomplete(messages=messages, model="deepseek-chat")

@pytest.mark.asyncio
async def test_invalid_model(provider):
    messages = [Message(role=Role.USER, content="Hello")]
    
    with pytest.raises(ProviderError):
        await provider.acomplete(messages=messages, model="invalid-model")

@pytest.mark.asyncio
async def test_system_message_handling(provider):
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that always responds with 'Hello'"),
        Message(role=Role.USER, content="Say something")
    ]

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content="Hello"))]
    mock_response.usage = AsyncMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_response.id = "test-id"
    mock_response.object = "chat.completion"
    mock_response.created = 1234567890
    mock_response.model = "deepseek-chat"

    with patch.object(provider._async_client.chat.completions, 'create', return_value=mock_response):
        response = await provider.acomplete(
            messages=messages,
            model="deepseek-chat",
            temperature=0
        )
        
        assert isinstance(response, ModelResponse)
        assert "hello" in response.content.lower() 