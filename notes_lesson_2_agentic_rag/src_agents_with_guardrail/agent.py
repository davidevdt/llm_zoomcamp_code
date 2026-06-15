import json
from collections.abc import Callable
from typing import Any, Protocol

from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall


ToolDefinition = tuple[Callable[..., Any], dict[str, Any]]


class RunnableAgent(Protocol): 
    async def run(self, question: str) -> str: 
        ... 


class Agent(RunnableAgent): 
    def __init__(
        self, 
        openai_client: AsyncOpenAI, 
        tool_definitions: list[ToolDefinition], 
        instructions: str, 
        model: str, 
    ) -> None: 
        self.openai_client = openai_client 
        self.instructions = instructions 
        self.model = model 

        self.functions: dict[str, Callable[..., Any]] = {} 
        self.tool_schemas: list[dict[str, Any]] = [] 

        for function, schema in tool_definitions: 
            name = schema["name"] 

            if function.__name__ != name: 
                raise ValueError("Function name and tool schema name do not match.") 
            
            self.functions[name] = function         # called from Python
            self.tool_schemas.append(schema)        # goes to OpenAI 


    def call_tool(self, call: ResponseFunctionToolCall) -> dict[str, str]: 
        args = json.loads(call.arguments)
        function = self.functions[call.name]
        result = function(**args)

        return {
            "type": "function_call_output", 
            "call_id": call.call_id, 
            "output": json.dumps(result), 
        }
    

    async def run(self, question:str) -> str: 
        try: 
            messages = [
                {"role": "developer", "content": self.instructions}, 
                {"role": "user", "content": question}, 
            ]  

            while True: 
                response = await self.openai_client.responses.create(
                    model=self.model, 
                    input=messages, 
                    tools=self.tool_schemas, 
                )

                messages.extend(response.output) 
                has_function_calls = False 

                for entry in response.output: 
                    if entry.type == "function_call": 
                        print("function_call:", entry.name, entry.arguments)
                        result = self.call_tool(entry) 
                        messages.append(result) 
                        has_function_calls = True 
                
                if not has_function_calls: 
                    return response.output_text 
        
        except Exception as e: 
            print("ERROR:", repr(e))
            raise
                    
