# Agent Definition

In Legion, agents are the fundamental building blocks of your multi-agent system. This guide explains how to define and customize agents.

## Basic Agent Structure

An agent in Legion is defined using the `@agent` decorator:

```python
from legion.agents import agent
from legion.interface.decorators import tool

@agent(model="openai:gpt-4-turbo", temperature=0.2)
class MyAgent:
    """System prompt that defines the agent's role and behavior"""
    
    @tool
    def my_tool(self, param: str) -> str:
        """Tool description"""
        return f"Processed: {param}"
```

## Agent Decorator Parameters

- `model`: The LLM provider and model to use (e.g., "openai:gpt-4-turbo", "anthropic:claude-3")
- `temperature`: Controls randomness in responses (0.0 to 1.0)
- `tools`: List of tools available to the agent
- `memory`: Memory provider for storing conversation history

## System Prompts

The class docstring serves as the agent's system prompt:

```python
@agent(model="openai:gpt-4-turbo")
class AnalysisAgent:
    """You are an expert data analyst skilled in interpreting complex datasets.
    Always provide clear explanations and cite relevant statistics."""
```

## Adding Tools

Tools can be added in two ways:

1. As class methods:
```python
@agent(model="openai:gpt-4-turbo")
class CalculatorAgent:
    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers together"""
        return a + b
```

2. As external functions:
```python
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

@agent(model="openai:gpt-4-turbo", tools=[multiply])
class MathAgent:
    """A mathematical agent with access to multiplication"""
```

## Using Type Hints

Legion uses type hints for input validation:

```python
from typing import List, Optional
from pydantic import Field

@agent(model="openai:gpt-4-turbo")
class DataProcessor:
    @tool
    def process_items(
        self,
        items: List[str],
        prefix: Optional[str] = None,
        max_items: int = Field(default=10, gt=0, le=100)
    ) -> List[str]:
        """Process a list of items with optional prefix"""
        results = items[:max_items]
        if prefix:
            results = [f"{prefix}: {item}" for item in results]
        return results
```

## Best Practices

1. **Clear System Prompts**: Write detailed, specific system prompts
2. **Tool Documentation**: Always provide clear docstrings for tools
3. **Type Safety**: Use type hints and Pydantic fields for validation
4. **Modular Design**: Keep agents focused on specific tasks
5. **Error Handling**: Implement proper error handling in tools

## Next Steps

- Learn about [System Prompts](system-prompts.md)
- Understand [Agent State](agent-state.md)
- Explore [Agent Communication](agent-communication.md)
