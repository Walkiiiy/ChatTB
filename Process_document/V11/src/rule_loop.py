import os
import sqlite3

from dotenv import load_dotenv

from tqdm import tqdm

from openai import OpenAI

from mcp.client.stdio import stdio_client

import asyncio

import json

import copy

load_dotenv()

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"

class RuleProcesser:
    def __init__(self,
                 load_path,
                 dump_path,
                 db_root_path,
                 reasonMCPserver_path,
                 testMCPserver_path,
                 reasonPrompt_path,
                 ):
        
        self.load_path=load_path
        self.dump_path=dump_path
        self.db_root_path=db_root_path
        
        with open(reasonPrompt_path) as f:
            self.resonPrompt=f.read()
        # with open(rulePrompt) as f:
        #     self.rulePrompt=f.read()
        with open(self.load_path)as f:
            self.evalRes=json.load(f)
        
        self.testClient=MCPClient(testMCPserver_path)
        self.reasonClient=MCPClient(reasonMCPserver_path)
        
        self.evalID="0"

    def generate_comment_prompt_testReason(self, question, Reason=None):
        base = "-- Using valid SQLite "
        base += " and the rules that explain the question and schema" if Reason else ""
        base += ", solve the following question by generating SQLite query for the tables provided above."
        prompt = f"{'-- latent mistakes: \n' + Reason if Reason else ''}\n{base}\n-- {question}"
        return prompt

    def generate_schema_prompt(self,db_path, num_rows=None):
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
                rows_prompt = self.nice_look_table(column_names=column_names, values=values)
                verbose_prompt = f"/* \n {num_rows} example rows: \n SELECT * FROM {cur_table} LIMIT {num_rows}; \n {rows_prompt} \n */"
                schemas[table[0]] = f"{create_prompt} \n {verbose_prompt}"
        conn.close()
        for v in schemas.values():
            full_schema_prompt_list.append(v)
        return "\n\n".join(full_schema_prompt_list)

    def nice_look_table(self,column_names: list, values: list):
        rows = []
        widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
        header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
        for value in values:
            row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
            rows.append(row)
        return header + '\n' + '\n'.join(rows)


    def evaluate_res(self,evalObj):
        predicted_sql=evalObj['sql']
        ground_truth=evalObj['ground_truth']
        db_path=os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite')
        
        res=self.execute_sql(predicted_sql,ground_truth,db_path)
        
        evalObj['res']=res
        return res
    
    def execute_sql(self,predicted_sql,ground_truth, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        try:
            cursor.execute(predicted_sql)
            predicted_res = cursor.fetchall()
        except Exception:
            return 0
        res = 0
        if set(predicted_res) == set(ground_truth_res):#####
            res = 1
        return res

    def getPrompt_testReason(self,evalObj):
        systemPrompt='''
            You are a helpful assistant that writes valid SQLite queries based on provided schemas.
            you will be given a database's schema and a question and some latent mistakes you should avoid, 
            you should combine the schema, think step by step and carefully to solve the given question, 
            then generate a SQLite query that solve the question best.
            you have one function, you have to call it.
            function SQLite_receiver takes the final SQLite query you generate.
            '''
        schema_prompt=self.generate_schema_prompt(os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite'))
        comment_prompt=self.generate_comment_prompt_testReason(evalObj['question'],"\n".join(evalObj['reason']))
        Prompt=f"{schema_prompt}\n\n{comment_prompt}, \nGenerate the SQL after thinking step by step"
        return systemPrompt,Prompt
    
    def getPrompt_getReason(self,evalObj):
        systemPrompt=self.resonPrompt
        # no schmea input while generating reason
        # schema_prompt=self.generate_schema_prompt(os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite'))
        Prompt=f'''
        the question\n:
        {evalObj['question']}'\n'
        {'the Reasons found before:\n'
         +'\n'.join(evalObj['reason']) if evalObj['reason'] else ''}
        the wrong sql\n:
        {evalObj['sql']}'\n'
        the target right sql\n:
        {evalObj['ground_truth']}
        '''

        return systemPrompt,Prompt

    def getReason(self,evalObj):
        systemPrompt,prompt=self.getPrompt_getReason(evalObj)
        res=self.reasonClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        if 'error' in res:
            return res            
        evalObj['reason']+=res['reason']
        return res

    def testReason(self,evalObj):
        systemPrompt,prompt=self.getPrompt_testReason(evalObj)
        res=self.testClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        if 'error' in res:
            return res 
        evalObj['sql']=res['sql']
        return res


    def reason_loop(self,maxRefineloop=3):
        totalCount=0
        correctCount=0
        for self.evalID, originObj in tqdm(self.evalRes.items(),total=len(self.evalRes),desc="Processing"):
            totalCount+=1
            evalObj=copy.deepcopy(originObj)
            if evalObj['res']==1:
                print(f'question{self.evalID} is able to output right result, skip.')
                correctCount+=1
                continue
            if len(evalObj['reason'])>=maxRefineloop:
                print(f'question{self.evalID} has failed before,skip.')
                continue
            print("processing question: ",self.evalID,' ',evalObj['question'])
            loop_count=0
            while evalObj['res']==0:
                print('round',loop_count+1)
                if len(evalObj['reason'])>1 and evalObj['reason'][-1]==evalObj['reason'][-2]:# if identical Reason was generated, probably fail.
                    print(f"identical reason generated, giving up.")
                    break
                self.getReason(evalObj)
                self.testReason(evalObj)
                self.evaluate_res(evalObj)
                loop_count+=1
                if evalObj['res']==1:
                    print(f"{GREEN}Success!{RESET}corrected within {loop_count} loop")
                    correctCount+=1
                if evalObj['res']==0 and loop_count>=maxRefineloop:
                    print(f"{RED}fail:{RESET}could not refine Reason of object in {maxRefineloop} loops, giving up.")
                    break
            self.evalRes[self.evalID]=copy.deepcopy(evalObj)
            json.dump(self.evalRes,open(self.dump_path,'w'),indent=4)
            
            print('ex: ',correctCount/totalCount,'\n----------------------------\n')


class MCPClient:
    def __init__(self, server_path: str):
        self.client = OpenAI()
        self.server_path = server_path
        self.available_tools = []
        self._load_tools_once()

    def _load_tools_once(self) -> None:
        asyncio.run(self._load_tools())

    async def _load_tools(self) -> None:
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command="uv",
            args=["run", self.server_path],
            env=None,
        )

        async with AsyncExitStack() as stack:
            stdio, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(stdio, write))

            await session.initialize()
            response = await session.list_tools()

            self.available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    },
                }
            for tool in response.tools
            ]

    def connect_gpt(
        self,
        systemPrompt: str,
        prompt: str,
        engine: str = "deepseek-chat",
        max_tokens: int = 512,
        temperature: float = 0,
        stop=None,
    ):
        result = self.client.chat.completions.create(
            model=engine,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": prompt},
            ],
            tools=self.available_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        choice = result.choices[0]
        # print(choice)
        if getattr(choice, "finish_reason", None) == "tool_calls":
            res = {}
            for tc in choice.message.tool_calls:
                res.update(json.loads(tc.function.arguments))
            return res
        return {"error": "something went wrong, tool did not call."}


load_path='/home/walkiiiy/ChatTB/Process_document/V11/part_3.json'
dump_path='/home/walkiiiy/ChatTB/Process_document/V11/part_3.json'
db_root_path='/home/walkiiiy/ChatTB/Bird_dev/dev_databases'
getReasonMCPserver_path='/home/walkiiiy/ChatTB/Process_document/V11/src/MCPserver_getReasons.py'
testMCPserver_path='/home/walkiiiy/ChatTB/Process_document/V11/src/MCPserver_testSql.py'
reasonPrompt='/home/walkiiiy/ChatTB/Process_document/V11/prompts/getReason.md'

Processer=RuleProcesser(load_path=load_path,
                            dump_path=dump_path,
                            db_root_path=db_root_path,
                            reasonMCPserver_path=getReasonMCPserver_path,
                            testMCPserver_path=testMCPserver_path,
                            reasonPrompt_path=reasonPrompt,
                            )

Processer.reason_loop(maxRefineloop=3)