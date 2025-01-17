"""Tests for the Gemini provider"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from legion.errors import ProviderError
from legion.interface.schemas import Message, ProviderConfig, Role
from legion.interface.tools import BaseTool
from legion.providers.gemini import GeminiFactory, GeminiProvider

# Load environment variables
load_dotenv()

class MockResponse:
    """Mock response for testing"""

    def __init__(self, content: str, tool_calls=None):
        self.id = "mock-id"
        self.object = "chat.completion"
        self.created = 1234567890
        self.model = "gemini-1.5-pro"
        self.choices = [
            MagicMock(
                index=0,
                message=MagicMock(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls if tool_calls else None
                )
            )
        ]
        self.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )

    @property
    def choices(self):
        return self._choices

    @choices.setter
    def choices(self, value):
        self._choices = value


class TestGeminiFactory:
    """Test the Gemini provider factory"""

    def test_create_provider(self):
        """Test creating a provider instance"""
        factory = GeminiFactory()
        provider = factory.create_provider()
        assert isinstance(provider, GeminiProvider)


class TestGeminiProvider:
    """Test the Gemini provider implementation"""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing"""
        with patch("openai.AsyncOpenAI") as mock:
            mock_instance = AsyncMock()
            mock.return_value = mock_instance
            mock_instance.chat = AsyncMock()
            mock_instance.chat.completions = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock()

            config = ProviderConfig(api_key="test-key")
            provider = GeminiProvider(config=config, debug=True)
            provider._setup_client()  # Initialize with mock
            provider.client = mock_instance  # Ensure mock is set
            return provider

    def test_setup_client_no_api_key(self):
        """Test setup fails without API key"""
        # Remove any environment variable
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        with pytest.raises(ProviderError, match="API key is required"):
            provider = GeminiProvider(config=ProviderConfig())
            provider._setup_client()

    def test_setup_client_with_env_key(self):
        """Test setup with environment API key"""
        with patch("openai.AsyncOpenAI") as mock:
            os.environ["GEMINI_API_KEY"] = "test-key"
            GeminiProvider(config=ProviderConfig())
            mock.assert_called_once_with(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=60,
                max_retries=3
            )
            del os.environ["GEMINI_API_KEY"]

    @pytest.mark.asyncio
    async def test_chat_completion(self, provider):
        """Test basic chat completion"""
        response = MockResponse("Test response")
        provider.client.chat.completions.create.return_value = response

        messages = [Message(role=Role.USER, content="Test message")]
        result = await provider._aget_chat_completion(
            messages=messages,
            model="gemini-1.5-pro",
            temperature=0.7
        )

        assert result.content == "Test response"
        assert result.usage.total_tokens == 30
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_tool_completion(self, provider):
        """Test completion with tool usage"""
        # First response with tool call
        function_mock = MagicMock()
        function_mock.name = "test_tool"  # Set as string instead of MagicMock
        function_mock.arguments = '{"arg": "value"}'

        tool_call = MagicMock(
            id="call-123",
            type="function",
            function=function_mock
        )
        first_response = MockResponse("", [tool_call])  # Empty content since using tool
        final_response = MockResponse("Final response after tool use")  # Response after tool use
        
        # Set up responses sequence
        provider.client.chat.completions.create.side_effect = [
            first_response,  # First call returns tool call
            final_response  # Second call returns final response
        ]

        # Create test tool
        class TestParameters(BaseModel):
            arg: str

        class TestTool(BaseTool):
            """Test tool for testing"""

            def __init__(self):
                super().__init__(
                    name="test_tool",
                    description="A test tool",
                    parameters=TestParameters
                )

            def run(self, arg: str) -> str:
                """Implement the sync run method"""
                return f"Tool result: {arg}"

            async def arun(self, arg: str) -> str:
                """Implement the async run method"""
                return self.run(arg)  # Reuse sync implementation

        messages = [Message(role=Role.USER, content="Use the tool")]
        tools = [TestTool()]

        response = await provider._aget_tool_completion(
            messages=messages,
            model="gemini-1.5-pro",
            tools=tools,
            temperature=0.7
        )

        assert response.content == "Final response after tool use"
        assert response.usage.total_tokens == 30
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_json_completion(self, provider):
        """Test JSON completion"""
        class TestSchema(BaseModel):
            name: str
            value: int

        response = MockResponse('{"name": "test", "value": 42}')
        provider.client.chat.completions.create.return_value = response

        messages = [Message(role=Role.USER, content="Get JSON")]
        result = await provider._aget_json_completion(
            messages=messages,
            model="gemini-1.5-pro",
            schema=TestSchema,
            temperature=0.7
        )

        assert json.loads(result.content) == {"name": "test", "value": 42}
        assert result.usage.total_tokens == 30

    def test_validate_request(self, provider):
        """Test request parameter validation"""
        # Test temperature=0 handling
        kwargs = provider._validate_request({"temperature": 0})
        assert kwargs["temperature"] == 0

        # Test max_tokens handling
        kwargs = provider._validate_request({"max_tokens": 100})
        assert kwargs["max_tokens"] == 100

        # Test invalid temperature
        with pytest.raises(ProviderError):
            provider._validate_request({"temperature": 2.0})

        # Test invalid max_tokens
        with pytest.raises(ProviderError):
            provider._validate_request({"max_tokens": 0})

        # Test unsupported parameters are removed
        kwargs = provider._validate_request({
            "temperature": 0.5,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "top_logprobs": None
        })
        assert "top_p" not in kwargs
        assert "frequency_penalty" not in kwargs
        assert "presence_penalty" not in kwargs
        assert "top_logprobs" not in kwargs
        assert kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_tool_and_json_completion(self, provider):
        """Test combined tool usage and JSON formatting"""
        # Define test schema
        class TestSchema(BaseModel):
            name: str
            age: int
            hobbies: List[str]

        # First response with tool call
        function_mock = MagicMock()
        function_mock.name = "test_tool"
        function_mock.arguments = '{"arg": "hello world"}'
        
        tool_call = MagicMock(
            id="call-123",
            type="function",
            function=function_mock
        )

        # Set up response sequence
        first_response = MockResponse("", [tool_call])  # Tool call response
        json_response = MockResponse('{"name": "Alice", "age": 30, "hobbies": ["dancing", "singing"]}')  # Final JSON response
        
        provider.client.chat.completions.create.side_effect = [
            first_response,  # First call returns tool call
            json_response  # Second call returns JSON response
        ]

        # Create test tool
        class TestParameters(BaseModel):
            arg: str

        class TestTool(BaseTool):
            """Test tool for testing"""
            def __init__(self):
                super().__init__(
                    name="test_tool",
                    description="A test tool",
                    parameters=TestParameters
                )

            def run(self, arg: str) -> str:
                """Implement the sync run method"""
                return f"Tool result: {arg}"

            async def arun(self, arg: str) -> str:
                """Implement the async run method"""
                return self.run(arg)

        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools when appropriate."),
            Message(
                role=Role.USER,
                content="First use the test_tool with the message 'hello world', then create a JSON response about a person named Alice who is 30 and likes dancing and singing."
            )
        ]

        response = await provider._aget_tool_completion(
            messages=messages,
            model="gemini-1.5-pro",
            tools=[TestTool()],
            temperature=0,
            format_json=True,
            json_schema=TestSchema
        )

        # Verify tool usage
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "test_tool"
        assert response.tool_calls[0]["function"]["arguments"] == '{"arg": "hello world"}'

        # Verify JSON response
        json_data = json.loads(response.content)
        assert json_data["name"] == "Alice"
        assert json_data["age"] == 30
        assert "dancing" in json_data["hobbies"]
        assert "singing" in json_data["hobbies"]


if __name__ == "__main__":
    pytest.main([__file__])
