#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from typing import Dict
from typing import Optional
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI, OpenAIError, RateLimitError
from openai import OpenAI


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import asyncio

load_dotenv()
SERVER_FILE_PATH='/home/walkiiiy/ChatTB/Evaluation/src/MCPserver.py'


SAVE_EVERY = 1

def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def nice_look_table(column_names: list, values: list):
    rows = []
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    return header + '\n' + '\n'.join(rows)

def generate_schema_prompt(db_path, num_rows=None):
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table[0] == 'sqlite_sequence':
            continue
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table[0]}';")
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = f"`{table[0]}`" if table[0] in ['order', 'by', 'group'] else table[0]
            cursor.execute(f"SELECT * FROM {cur_table} LIMIT {num_rows}")
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = f"/* \n {num_rows} example rows: \n SELECT * FROM {cur_table} LIMIT {num_rows}; \n {rows_prompt} \n */"
            schemas[table[0]] = f"{create_prompt} \n {verbose_prompt}"
    conn.close()
    for v in schemas.values():
        full_schema_prompt_list.append(v)
    return "\n\n".join(full_schema_prompt_list)

def generate_comment_prompt(question, knowledge=None):
    base = "-- Using valid SQLite"
    base += " and understanding External Knowledge" if knowledge else ""
    base += ", solve the following question by generating SQLite query for the tables provided above."
    prompt = f"{'-- External Knowledge: ' + knowledge if knowledge else ''}\n{base}\n-- {question}"
    return prompt

def cot_wizard():
    return "\nGenerate the SQL after thinking step by step:"

def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path)
    comment_prompt = generate_comment_prompt(question, knowledge)
    return f"{schema_prompt}\n\n{comment_prompt}{cot_wizard()}\nSELECT "

def quota_giveup(e):
    return isinstance(e, RateLimitError) and "quota" in str(e)

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI()
        self.avalible_tools=[]
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

        response = await self.session.list_tools()
        # 生成 function call 的描述信息
        self.available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]

    async def connect_gpt(self,engine, prompt, max_tokens, temperature, stop=None):
        # 获取所有 mcp 服务器 工具列表信息
        
        systemPrompt='''
            You are a helpful assistant that writes valid SQLite queries based on provided schemas.
            you will be given a database's schema and a question, 
            you should combine the schema, think step by step and carefully to solve the given question, 
            then generate a SQLite query that solve the question best.
            you have two functions, you have to call both of them.
            function reasoning_receiver takes the reasoning process you analyze and solve the question, 
            function SQLite_receiver takes the final SQLite query you generate.
            '''

        result = self.client.chat.completions.create(
            model=engine,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": prompt}
            ],
            tools=self.available_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop
        )
        content = result.choices[0]
        if content.finish_reason == "tool_calls":
            # 如何是需要使用工具，就解析工具
            # Iterate over all the tool calls
            tool_calls = content.message.tool_calls
            res={}
            for tool_call in tool_calls:
                # tool_name = tool_call.function.name
                res.update(json.loads(tool_call.function.arguments))
            return res
        
        print('something went wrong, tool did not call.')
        return {'error':'something went wrong, tool did not call.'}
    

    async def collect_response_from_gpt(self,db_path_list, question_list,engine, knowledge_list=None, output_path=None):
        responses = {}

        # Resume if file already exists
        if output_path and os.path.exists(output_path):
            with open(output_path, 'r') as f:
                responses = json.load(f)
            print(f"🔁 Resuming from previous progress: {len(responses)} items already processed.")

        start_idx = len(responses)
        for i in tqdm(range(start_idx, len(question_list))):
            print(f"\n--- Processing {i + 1}/{len(question_list)} ---")
            print(f"Question: {question_list[i]}")
            prompt = generate_combined_prompts_one(
                db_path=db_path_list[i],
                question=question_list[i],
                knowledge=knowledge_list[i] if knowledge_list else None
            )
            result = await self.connect_gpt(engine=engine, prompt=prompt, max_tokens=512, temperature=0)
            result['question']=question_list[i]
            db_id = os.path.basename(db_path_list[i]).replace('.sqlite', '')
            result['db_id']=db_id
            # sql = sql + f'\t----- bird -----\t{db_id}'
            responses[str(i)] = result
            # ✅ Save every N steps
            if output_path and ((i + 1) % SAVE_EVERY == 0 or i == len(question_list) - 1):
                new_directory(os.path.dirname(output_path))
                with open(output_path, 'w') as f:
                    json.dump(responses, f, indent=4)
                print(f"💾 Progress saved to {output_path}")

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for data in datasets:
        question_list.append(data['question'])
        db_path = os.path.join(db_root_path, data['db_id'], data['db_id'] + '.sqlite')
        db_path_list.append(db_path)
        knowledge_list.append(data.get('evidence', None))
    return question_list, db_path_list, knowledge_list

def generate_sql_file(sql_lst, output_path=None):
    result = {i: sql for i, sql in enumerate(sql_lst)}
    if output_path:
        new_directory(os.path.dirname(output_path))
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
    return result

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='dev')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--use_knowledge', type=str, default='False')
    parser.add_argument('--db_root_path', type=str, default='')
    parser.add_argument('--engine', type=str, required=True, default='gpt-4')
    parser.add_argument('--data_output_path', type=str)
    parser.add_argument('--chain_of_thought', type=str)
    args = parser.parse_args()

    eval_data = json.load(open(args.eval_path, 'r'))
    question_list, db_path_list, knowledge_list = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)

    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    if args.chain_of_thought == 'True':
        output_name = args.data_output_path + f'predict_{args.mode}_cot.json'
    else:
        output_name = args.data_output_path + f'predict_{args.mode}.json'
    client = MCPClient()
    await client.connect_to_server()

    if args.use_knowledge == 'True':
        await client.collect_response_from_gpt(
            db_path_list=db_path_list,
            question_list=question_list,
            engine=args.engine,
            knowledge_list=knowledge_list,
            output_path=output_name
        )
    else:
        await client.collect_response_from_gpt(
            db_path_list=db_path_list,
            question_list=question_list,
            engine=args.engine,
            knowledge_list=None,
            output_path=output_name
        )

    print(f'\n✅ Successfully collected results from {args.engine} for {args.mode} evaluation.')
    print(f'📁 Output saved to: {output_name}')
    print(f'🧠 Used knowledge: {args.use_knowledge}, 🧵 Chain of Thought: {args.chain_of_thought}')


if __name__ == '__main__':
    asyncio.run(main())