import os
import sqlite3
from typing import Optional
from contextlib import AsyncExitStack

from dotenv import load_dotenv

from tqdm import tqdm

from openai import OpenAI


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import asyncio

import json

import copy

load_dotenv()


class EvidenceProcesser:
    def __init__(self,predict_dev_eval_path,db_root_path,testMCPserver_path,getMCPserver_path,get_evidence_prompt_ver):
        self.get_evidence_prompt_path='/home/walkiiiy/ChatTB/Evaluation/prompts/'+'get_evidence'+get_evidence_prompt_ver+'.md'
        self.res_path=predict_dev_eval_path
        with open(predict_dev_eval_path)as f:
            self.evalRes=json.load(f)
        self.testEvidenceClient=MCPClient(testMCPserver_path)
        self.getEvidenceClient=MCPClient(getMCPserver_path)
        self.db_root_path=db_root_path
        self.evalID="0"

    def generate_comment_prompt(self, question, evidence=None):
        base = "-- Using valid SQLite "
        base += " and the tips that explain the question and schema" if evidence else ""
        base += ", solve the following question by generating SQLite query for the tables provided above."
        prompt = f"{'-- tips: ' + evidence if evidence else ''}\n{base}\n-- {question}"
        return prompt

    def generate_schema_prompt(self,db_path, num_rows=None):
        full_schema_prompt_list = []
        # print('connecting to database ',db_path)
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

    def gpt_request_testEvidence(self,evalObj):
        systemPrompt,prompt=self.getPrompt_testEvidence(evalObj)
        # print("\n\ntesting with prompt:\n",prompt)
        res=self.testEvidenceClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        # print("loop result sql:\n",res['sql'])
        if 'error' in res:
            return res 
        evalObj['sql']=res['sql']
        evalObj['reason']=res['reason']
        return res

    def gpt_request_getEvidence(self,evalObj):
        systemPrompt,prompt=self.getPrompt_getEvidence(evalObj)
        res=self.getEvidenceClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        if 'error' in res:
            return res            
        # print(res)
        evalObj['evidence'].append(res['evidence'])
        evalObj['reason']=res['reason']
        return res

    def evaluate_res(self,evalObj):
        predicted_sql=evalObj['sql']
        ground_truth=evalObj['ground_truth']
        # print("\nprediced:/n",predicted_sql,"correct sql:\n",ground_truth)
        db_path=os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite')
        
        res=self.execute_sql(predicted_sql,ground_truth,db_path)
        
        evalObj['res']=res
        # print('loop res: ',res)
        return res
    
    def execute_sql(self,predicted_sql,ground_truth, db_path):
        conn = sqlite3.connect(db_path)
        # Connect to the database
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

    def getPrompt_testEvidence(self,evalObj):
        systemPrompt='''
            You are a helpful assistant that writes valid SQLite queries based on provided schemas.
            you will be given a database's schema and a question, 
            you should combine the schema, think step by step and carefully to solve the given question, 
            then generate a SQLite query that solve the question best.
            you have two functions, you have to call both of them.
            function reasoning_receiver takes the reasoning process you analyze and solve the question, 
            function SQLite_receiver takes the final SQLite query you generate.
            '''
        schema_prompt=self.generate_schema_prompt(os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite'))
        comment_prompt=self.generate_comment_prompt(evalObj['question'],"\n".join(evalObj['evidence']))
        Prompt=f"{schema_prompt}\n\n{comment_prompt}, \nGenerate the SQL after thinking step by step"
        # print("\ngetting sql with prompt:",Prompt)
        return systemPrompt,Prompt
    
    def getPrompt_getEvidence(self,evalObj):
        with open(self.get_evidence_prompt_path) as f:
            systemPrompt=f.read()

        schema_prompt=self.generate_schema_prompt(os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite'))
        Prompt=f'''
        the schema of the database:
        {schema_prompt}
        the question:
        {evalObj['question']}
        the evidences found before:
        {evalObj['evidence']}
        the incorrect reasoning process you generated before:
        {evalObj['reason']}
        the wrong sql you generated before:
        {evalObj['sql']}
        the target right sql you supposed to generate:
        {evalObj['ground_truth']}
        '''

        return systemPrompt,Prompt

    def test_looped_effect(self,testRange):
        for self.evalID in tqdm(range(testRange)):
            evalObj=copy.deepcopy(self.evalRes[str(self.evalID)])
            print("processing question: ",self.evalID,' ',evalObj['question'])
            self.gpt_request_testEvidence(evalObj)
            self.evaluate_res(evalObj)
            self.evalRes[str(self.evalID)]=copy.deepcopy(evalObj)
            json.dump(self.evalRes,open(self.res_path,'w'),indent=4)


    def reason_loop(self,maxRefineloop=3):
        totalCount=0
        correctCount=0
        for self.evalID, originObj in tqdm(self.evalRes.items(),total=len(self.evalRes),desc="Processing"):
            totalCount+=1
            evalObj=copy.deepcopy(originObj)
            if evalObj['res']==1:
                print(f'question{self.evalID} has been refined, skip.')
                correctCount+=1
                continue
            if len(evalObj['evidence'])>=maxRefineloop:
                print(f'question{self.evalID} has failed before,skip.')
                continue
            # evalObj['evidence']=evalObj['evidence'][:1]## prevent similar hitorical evidence rolling
            print("processing question: ",self.evalID,' ',evalObj['question'])
            loop_count=0
            while evalObj['res']==0:
                print('round',loop_count+1)
                if len(evalObj['evidence'])>1 and evalObj['evidence'][-1]==evalObj['evidence'][-2]:# if identical evidence was generated, probably fail.
                    print(f"identical evidence generated, giving up.")
                    break
                self.gpt_request_getEvidence(evalObj)
                self.gpt_request_testEvidence(evalObj)
                self.evaluate_res(evalObj)
                loop_count+=1
                if evalObj['res']==1:
                    print(f"refined within {loop_count} loops")
                    correctCount+=1
                if evalObj['res']==0 and loop_count>=maxRefineloop:
                    print(f"could not refine evidence of object in {maxRefineloop} loops, giving up.")
                    break
            self.evalRes[self.evalID]=copy.deepcopy(evalObj)
            json.dump(self.evalRes,open(self.res_path,'w'),indent=4)
            
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
        # 走到这里，上下文已在“同一个任务”里完成关闭，不会再有跨任务退出的问题。

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


predict_dev_eval_path='/home/walkiiiy/ChatTB/Evaluation/exp_result/v10_output/part_1.json'
db_root_path='/home/walkiiiy/ChatTB/Bird_dev/dev_databases'
getMCPserver_path='/home/walkiiiy/ChatTB/Evaluation/src/MCPserver_getEvidence.py'
testMCPserver_path='/home/walkiiiy/ChatTB/Evaluation/src/MCPserver_testEvidence.py'
Processer=EvidenceProcesser(predict_dev_eval_path,
                            db_root_path,
                            testMCPserver_path,
                            getMCPserver_path,
                            get_evidence_prompt_ver='V4'
                            )

Processer.process_evidence_loop()

# Processer.test_looped_effect(150)

# res=Processer.gpt_request_testEvidence(Processer.evalRes['76'])
















# obj=Processer.evalRes['1']
# systemPrompt,prompt=Processer.getPrompt_getEvidence(obj)
# systemPrompt,prompt=Processer.getPrompt_testEvidence(obj)
# print(systemPrompt)
# print(prompt)
# sql1='''
# SELECT
# 1.0 * "Free Meal Count (Ages 5-17)" / "Enrollment (Ages 5-17)" AS eligible_free_rate
# FROM frpm
# WHERE
# "Educational Option Type" = 'Continuation School'
# AND "Free Meal Count (Ages 5-17)" IS NOT NULL
# AND "Enrollment (Ages 5-17)" IS NOT NULL
# AND "Enrollment (Ages 5-17)" > 0
# ORDER BY eligible_free_rate ASC
# LIMIT 3;
# '''
# sql2='''
# SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3'''
# res=Processer.execute_sql(sql1,sql2,'/home/walkiiiy/ChatTB/Bird_dev/dev_databases/california_schools/california_schools.sqlite')
# print(res)












# the old version of MCPClient, needs await out side
# class MCPClient:
#     def __init__(self,server_path):
#         self.session: Optional[ClientSession] = None
#         self.exit_stack = AsyncExitStack()
#         self.client = OpenAI()
#         self.avalible_tools=[]
#         self.server_path=server_path
#         self.connect_to_server()

#     async def connect_to_server(self):
#         server_params = StdioServerParameters(
#             command='uv',
#             args=['run', self.server_path],
#             env=None
#         )

#         stdio_transport = await self.exit_stack.enter_async_context(
#             stdio_client(server_params))
#         stdio, write = stdio_transport
#         self.session = await self.exit_stack.enter_async_context(
#             ClientSession(stdio, write))

#         await self.session.initialize()

#         response = await self.session.list_tools()
#         self.available_tools = [{
#             "type": "function",
#             "function": {
#                 "name": tool.name,
#                 "description": tool.description,
#                 "input_schema": tool.inputSchema
#             }
#         } for tool in response.tools]

#     async def connect_gpt(self,systemPrompt,prompt,\
#                           engine='deepseek-chat',max_tokens=512, temperature=0, stop=None):
#         result = self.client.chat.completions.create(
#             model=engine,
#             messages=[
#                 {"role": "system", "content": systemPrompt},
#                 {"role": "user", "content": prompt}
#             ],
#             tools=self.available_tools,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             stop=stop
#         )
#         content = result.choices[0]
#         if content.finish_reason == "tool_calls":
#             tool_calls = content.message.tool_calls
#             res={}
#             for tool_call in tool_calls:
#                 # tool_name = tool_call.function.name
#                 res.update(json.loads(tool_call.function.arguments))
#             return res
        
#         print('something went wrong, tool did not call.')
#         return {'error':'something went wrong, tool did not call.'}


