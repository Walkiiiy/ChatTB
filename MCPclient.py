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
DESCRIPTION=os.getenv("DEV_DESCRIPTION")

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
        # 这里需要通过 system prompt 来约束一下大语言模型，
        # 否则会出现不调用工具，自己乱回答的情况
        system_prompt = (
            "you are a helpful data analyze assistant"
            "you will be given a database schema, which contains the specific description of all the columns of all the tables in the database"
            "please analyze which column name or value is ambiguous, may cause trouble in LLM's understanding off the schema."
            "you have the a function wiki_search to look up the meanings in wikipedia, which can take all the query from a schema, returning documents"
            "if the value meaning seems ambiguous, and the type of it's column is discrete instead of continuous, use wiki_search function to look up its meaning."
            "if the column name seems ambiguous, use wiki_search function to look up its meaning."
            "please combine schema information like database name, table name, column description, target ambiguous word to infer some all the possible and understandable full-name queries when using wiki_search."
        )
        
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
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # 执行工具
            result = await self.session.call_tool(tool_name, tool_args)
            print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")
			
            # # 将 deepseek 返回的调用哪个工具数据和工具执行完成后的数据都存入messages中
            # messages.append(content.message.model_dump())
            # messages.append({
            #     "role": "tool",
            #     "content": result.content[0].text,
            #     "tool_call_id": tool_call.id,
            # })
            # print('messages:',messages)
            # # 将上面的结果再返回给 deepseek 用于生产最终的结果
            # response = self.client.chat.completions.create(
            #     model=os.getenv("OPENAI_MODEL"),
            #     messages=messages,
            # )
            # return response.choices[0].message.content

            return result.content[0].text

        return content.message.content
        
    async def process_loop(self):
        # while True:
        # query = input("\nQuery: ").strip()

        # if query.lower() == 'quit':
        #     break
        for schema_doc in os.listdir(DESCRIPTION):
            with open(os.path.join(DESCRIPTION,schema_doc),'r')as f:
                schema=json.load(f)
            response = await self.process_schema(schema)
            print("\n" + response)

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
