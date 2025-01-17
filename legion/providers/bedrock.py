import json
from typing import Any, Dict, List, Optional, Sequence, Type, Tuple

import boto3
from pydantic import BaseModel, ValidationError

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


class BedrockFactory(ProviderFactory):
    """Factory for creating AWS Bedrock providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new AWS Bedrock provider instance"""
        return BedrockProvider(config=config or ProviderConfig(), **kwargs)


class BedrockProvider(LLMInterface):
    """AWS Bedrock-specific implementation of the LLM interface.

    Key Implementation Details:
    1. Role Alternation: Bedrock strictly requires messages to alternate between 'user' and 'assistant' roles.
       System messages are handled separately in the request body's 'system' field.
    
    2. Tool Usage: Tools are implemented using Bedrock's native toolConfig format. Tool responses are
       formatted as user messages to maintain the required role alternation.
    
    3. JSON Formatting: When requesting JSON output, especially after tool usage, the conversation is
       reset with fresh system and user messages to ensure proper role alternation and clean context.
    
    4. Message Formatting: Messages are formatted into Bedrock's content blocks structure, where each
       block can contain either text content or tool usage information.

    Implementation Considerations:
    - Tool and JSON combinations require careful handling of conversation flow to maintain role alternation
    - System messages are extracted and sent separately in the request
    - Tool results are formatted as user messages to maintain the conversation flow
    - JSON formatting after tool usage requires a conversation reset to ensure clean output
    """

    def __init__(self, config: ProviderConfig, debug: bool = False):
        """Initialize provider with boto3 client"""
        super().__init__(config, debug)
        self._session = None
        self._client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """Initialize AWS Bedrock client"""
        try:
            self._session = boto3.Session(
                aws_access_key_id=self.config.api_key,
                aws_secret_access_key=self.config.api_secret,
                region_name=self.config.region or "us-east-1"
            )
            self._client = self._session.client(service_name='bedrock-runtime')
        except Exception as e:
            raise ProviderError(f"Failed to initialize AWS Bedrock client: {str(e)}")

    async def _asetup_client(self) -> None:
        """Initialize AWS Bedrock client asynchronously"""
        # AWS Bedrock doesn't have an async client, so we'll use the sync one
        self._setup_client()

    def _format_messages(self, messages: List[Message]) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, str]]]]:
        """Format messages for Bedrock API.
        
        Bedrock has specific requirements for message formatting:
        1. Messages must strictly alternate between user and assistant roles
        2. System messages are handled separately and returned as a separate list
        3. Content is formatted as blocks that can contain either text or tool usage
        4. Tool calls are formatted using Bedrock's toolUse structure
        
        Returns:
            Tuple containing:
            - List of formatted messages with alternating roles
            - Optional list of system messages (if any)
        """
        formatted_messages = []
        system_messages = []

        for message in messages:
            if message.role == Role.SYSTEM:
                system_messages.append({"text": message.content})
                continue

            msg_content = [{"text": message.content}]
            
            # Handle tool calls from assistant
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    msg_content.append({
                        "toolUse": {
                            "toolUseId": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"])
                        }
                    })

            formatted_message = {
                "role": message.role.value,
                "content": msg_content
            }

            formatted_messages.append(formatted_message)

        return formatted_messages, system_messages if system_messages else None

    def _format_tools(self, tools: Sequence[BaseTool]) -> List[Dict[str, Any]]:
        """Format tools for Bedrock API"""
        formatted_tools = []
        for tool in tools:
            # Get the schema from the tool's parameters
            if hasattr(tool.parameters, "model_json_schema"):
                schema = tool.parameters.model_json_schema()
                # Convert schema to match Bedrock's expected format
                input_schema = {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", [])
                }
            else:
                input_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

            # Format the tool spec according to Bedrock's requirements
            tool_spec = {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "json": input_schema
                    }
                }
            }
            formatted_tools.append(tool_spec)
        
        return formatted_tools

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response"""
        try:
            if not response.get('output'):
                return ""
            
            message = response['output']['message']
            content = message.get('content', [])
            
            # Extract text from all content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if 'text' in block:
                        text_parts.append(block['text'])
                    elif 'toolUse' in block:
                        # Skip tool use blocks
                        continue
                    elif 'json' in block:
                        # Return JSON content directly
                        return json.dumps(block['json'])
            
            return ' '.join(text_parts)
        except Exception as e:
            raise ProviderError(f"Failed to extract content from response: {str(e)}")

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from the response"""
        try:
            if not response.get('output', {}).get('message', {}).get('content'):
                return None

            response_content_blocks = response['output']['message']['content']
            tool_calls = []

            # Find blocks containing toolUse
            for block in response_content_blocks:
                if isinstance(block, dict) and 'toolUse' in block:
                    tool_use = block['toolUse']
                    tool_calls.append({
                        "id": tool_use.get('toolUseId', ''),
                        "type": "function",
                        "function": {
                            "name": tool_use.get('name', ''),
                            "arguments": json.dumps(tool_use.get('input', {}))
                        }
                    })

            return tool_calls if tool_calls else None
        except Exception as e:
            if self.debug:
                print(f"Failed to extract tool calls: {str(e)}")
            return None

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from the response"""
        try:
            usage = response.get('usage', {})
            return TokenUsage(
                prompt_tokens=usage.get('inputTokens', 0),
                completion_tokens=usage.get('outputTokens', 0),
                total_tokens=usage.get('totalTokens', 0)
            )
        except Exception:
            return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion synchronously"""
        try:
            chat_messages, system = self._format_messages(messages)

            request_body = {
                "modelId": model,
                "messages": chat_messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens or 2000,
                    "topP": 0.1,
                    "temperature": temperature
                },
                "additionalModelRequestFields": {
                    "inferenceConfig": {
                        "topK": 20
                    }
                }
            }
            if system:
                request_body["system"] = system

            response = self._client.converse(**request_body)

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"AWS Bedrock completion failed: {str(e)}")

    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion asynchronously"""
        # AWS Bedrock doesn't have async support, so we'll use the sync version
        return self._get_chat_completion(messages, model, temperature, max_tokens)

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
        """Get a tool-enabled chat completion synchronously.
        
        Implementation Details:
        1. Tool Usage Flow:
           - Tools are formatted according to Bedrock's toolConfig specification
           - Tool results are added as user messages to maintain role alternation
           - Assistant responses with tool calls are preserved in the conversation
        
        2. JSON Formatting:
           - When format_json is True, adds JSON schema instructions to system message
           - After tool usage, if JSON output is invalid, resets conversation with fresh
             system and user messages to ensure clean JSON formatting
           - Preserves tool calls history while resetting conversation context
        
        3. Role Alternation:
           - Maintains strict user/assistant alternation as required by Bedrock
           - System messages are handled separately in the request
           - Tool results are formatted as user messages to maintain alternation
        """
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
                chat_messages, system = self._format_messages(current_messages)
                tool_list = self._format_tools(tools)

                request_body = {
                    "modelId": model,
                    "messages": chat_messages,
                    "toolConfig": {
                        "tools": tool_list
                    },
                    "inferenceConfig": {
                        "maxTokens": max_tokens or 2000,
                        "topP": 0.1,
                        "temperature": temperature
                    },
                    "additionalModelRequestFields": {
                        "inferenceConfig": {
                            "topK": 20
                        }
                    }
                }
                if system:
                    request_body["system"] = system

                if self.debug:
                    print(f"\nSending request to Bedrock:")
                    print(f"Model: {model}")
                    print(f"Request body: {json.dumps(request_body, indent=2)}")

                response = self._client.converse(**request_body)

                if self.debug:
                    print(f"\nReceived response from Bedrock:")
                    print(f"Response: {json.dumps(response, indent=2)}")

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
                                raw_response=response,
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
                                content=f"Create a JSON object about a person with these details: {content}"
                            ))

                            return self._get_json_completion(
                                messages=json_messages,
                                model=model,
                                schema=json_schema,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                preserve_tool_calls=final_tool_calls
                            )

                    return ModelResponse(
                        content=content,
                        raw_response=response,
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
                        result = tool.run(**args)

                        # Add tool result as user message to maintain alternation
                        current_messages.append(Message(
                            role=Role.USER,
                            content=f"Tool '{tool_call['function']['name']}' returned: {json.dumps(result) if isinstance(result, dict) else str(result)}"
                        ))

            # If we've reached here, we hit max iterations
            return ModelResponse(
                content="Max iterations reached without completion",
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=final_tool_calls if final_tool_calls else None
            )

        except Exception as e:
            if self.debug:
                print(f"AWS Bedrock tool completion failed: {str(e)}")
            raise ProviderError(f"AWS Bedrock tool completion failed: {str(e)}")

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
        # AWS Bedrock doesn't have async support, so we'll use the sync version
        return self._get_tool_completion(
            messages=messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            format_json=format_json,
            json_schema=json_schema
        )

    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None,
        preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Get a JSON-formatted chat completion synchronously.
        
        Implementation Details:
        1. JSON Formatting:
           - Adds clear JSON schema instructions to guide the model's output
           - Validates response against the provided schema
           - Handles JSON parsing and validation errors with clear error messages
        
        2. Tool Integration:
           - Preserves tool calls from previous interactions if provided
           - Maintains clean context for JSON formatting while keeping tool history
        
        3. Role Management:
           - Ensures proper role alternation in the conversation
           - Handles system messages separately in the request structure
        """
        try:
            # Add JSON formatting instructions
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

            chat_messages, system = self._format_messages(current_messages)

            request_body = {
                "modelId": model,
                "messages": chat_messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens or 2000,
                    "topP": 0.1,
                    "temperature": temperature
                },
                "additionalModelRequestFields": {
                    "inferenceConfig": {
                        "topK": 20
                    }
                }
            }
            if system:
                request_body["system"] = system

            response = self._client.converse(**request_body)
            content = self._extract_content(response)

            # Try to parse the response as JSON
            try:
                json_data = json.loads(content)
                schema(**json_data)  # Validate against schema
            except (json.JSONDecodeError, ValidationError) as e:
                raise ProviderError(f"Failed to parse response as valid JSON: {str(e)}")

            return ModelResponse(
                content=content,
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=preserve_tool_calls
            )

        except Exception as e:
            if self.debug:
                print(f"AWS Bedrock JSON completion failed: {str(e)}")
            raise ProviderError(f"AWS Bedrock JSON completion failed: {str(e)}")

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
        # AWS Bedrock doesn't have async support, so we'll use the sync version
        return self._get_json_completion(messages, model, schema, temperature, max_tokens, preserve_tool_calls) 