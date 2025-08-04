import json
import asyncio
import os
from typing import Optional
from contextlib import AsyncExitStack

from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
SERVER_FILE_PATH=os.getenv("SERVER_FILE_PATH")
DESCRIPTION=os.getenv("TRAIN_DESCRIPTION")
DOCUMENT_PATH=os.getenv("TRAIN_DOCUMENT")


PromptFile='prompt_schemaAnalyze_V3_zeroShot.md'
with open(PromptFile,'r') as f:
    Prompt=f.read()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI()
    async def connect_to_server(self):
        server_params = StdioServerParameters(
            command='uv',
            args=['run', SERVER_FILE_PATH],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))

        await self.session.initialize()
    async def process_schema(self, schema: str) -> str:
        
        system_prompt = Prompt+'''
            you have two functions, you have to call both of them.
            function description_receiver takes the natural language schema description you generate, 
            function selected_word_receiver takes the selected words needs to supplement their background knowledge.
            '''

        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(schema)}
        ]

        # 获取所有 mcp 服务器 工具列表信息
        response = await self.session.list_tools()
        # 生成 function call 的描述信息
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]
        
        # 请求 deepseek，function call 的描述信息通过 tools 参数传入
        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=messages,
            tools=available_tools
        )

        # 处理返回的内容
        content = response.choices[0]
        if content.finish_reason == "tool_calls":
            # 如何是需要使用工具，就解析工具
            # Iterate over all the tool calls
            tool_calls = content.message.tool_calls
            
            # List to hold the async tasks if we are going to run them concurrently
            
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Execute the tool sequentially
                print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")
                result = await self.session.call_tool(tool_name, tool_args)
                
                # Optionally, print or process the result
                print(f"Result from {tool_name}: {result}")
			
            # return result.content[0].text
            return result

        return content
        
    async def process_loop(self):
        processed=[i.split('.')[0] for i in os.listdir(DOCUMENT_PATH)]
        for schema_doc in os.listdir(DESCRIPTION):
            if schema_doc.split('.')[0] in processed:
                print(f'{schema_doc} processed,skip.')
                continue
            print(f"processing {schema_doc}")
            with open(os.path.join(DESCRIPTION,schema_doc),'r')as f:
                schema=json.load(f)
            response = await self.process_schema(schema)
            print(schema_doc,' document generated.\n\n')

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.process_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
