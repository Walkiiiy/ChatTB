import os
import sqlite3

from dotenv import load_dotenv

from tqdm import tqdm

from openai import OpenAI

from mcp.client.stdio import stdio_client

import asyncio

import json

import copy

import warnings
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
                 solutionMCPserver_path,
                 testMCPserver_path,
                 ):
        
        self.load_path=load_path
        self.dump_path=dump_path
        self.db_root_path=db_root_path
        
        with open(self.load_path)as f:
            self.evalRes=json.load(f)

        self.testClient=MCPClient(testMCPserver_path)
        self.solutionClient=MCPClient(solutionMCPserver_path)
        
        self.evalID="0"

    def generate_comment_prompt_testSolution(self, question,solution):
        base = ""
        base += " using the given solution" 
        base += ", solve the following question by generating SQLite query."
        prompt = f"\n{base}\n-- question:\n{question} \n-- solution:\n{solution}"
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

    def getPrompt_testSolution(self,evalObj):
        systemPrompt='''
            You are a helpful assistant that writes valid SQLite queries based on provided schema and solution.
            '''
        schema_prompt=self.generate_schema_prompt(os.path.join(self.db_root_path, evalObj['db_id'], evalObj['db_id'] + '.sqlite'))
        comment_prompt=self.generate_comment_prompt_testSolution(evalObj['question'], {evalObj['solution'][-1] if len(evalObj['solution'])>0 else ''})
        Prompt=f'''
            you will be given a question related to a database and text solution of the question.
            the solution solves the question perfectly, all you need is to convert the solution into a valid SQLite query.
            you should generate a SQLite query that solve the question according to the given text solution.
            you have one function, you have to call it.
            function SQLite_receiver takes the final SQLite query you generate.
            \n{comment_prompt}, 
            \nyou have to call a function `SQLite_receiver` to return the SQL you generate'''
        return systemPrompt,Prompt
    
    def getPrompt_getSolution(self,evalObj):#actually a sql2text job
        systemPrompt='''
            - You are a helpful database assistant that convert the given sql to text operations of the database.
            '''
        prompt='''
            Describe how the given query operates as numbered steps **in one paragraph**. Follow these hard constraints:

            * **No SQL operation words** (e.g., select, where, group, count, order, limit, join).
            * Use the **exact table and column names** and **literal values** as written.
            * Write **step-by-step** using `1)... 2)... 3)...` within a single paragraph.
            * For **every derived value**, immediately append a **scope tag** in square brackets chosen from exactly these:

            * `[per-record]` (one row at a time)
            * `[per-bucket by <columns>]` (computed over records sharing the listed columns)
            * `[dataset after step N]` (computed over all records that remain after a specific earlier step)
            * `[entire table]` (ignores earlier narrowing steps)
            * Explicitly distinguish between values that are **shown in the final output** and values **used only to arrange the rows**.
            * Add a final step that **lists the output columns in order**, stating for each whether it is raw or derived and whether it is **the same for every output row**.
            * If only a subset of rows is kept, state **how many** and describe **tie behavior** (e.g., “if several have the same top value, keep any one of them”).
            * Do **not** invent columns or values; do **not** omit any value that appears in the final output.

            Example structure to follow (fill with the query’s concrete details):
            `1) Identify the source table(s) … 2) Keep only records where <column>=<value> … 3) Treat records with identical <columns> as one bucket … 4) Compute <measure A> [per-bucket by <columns>] … 5) Also compute <measure B> [dataset after step 2] … 6) Arrange buckets by <measure A> from largest to smallest (used only to arrange) … 7) Keep the first <K> buckets; if tied, keep any one … 8) Output columns: <col1/raw>, <col2/derived, same for every row>, …`        
            '''
        prompt+=f'''
        the sql\n:
        {evalObj['ground_truth']}
        '''

        return systemPrompt,prompt

    def getSolution(self,evalObj):
        systemPrompt,prompt=self.getPrompt_getSolution(evalObj)
        res=self.solutionClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        if 'error' in res:
            return res            
        evalObj['solution'].append(res['solution'])
        return res
        

    def testSolution(self,evalObj):
        systemPrompt,prompt=self.getPrompt_testSolution(evalObj)
        res=self.testClient.connect_gpt(systemPrompt,prompt,engine='deepseek-chat')
        if 'error' in res:
            evalObj['sql']='error'
            return res 
        evalObj['sql']=res['sql']
        return res


    def solution_loop(self,maxRefineloop=5):
        totalCount=0
        correctCount=0
        for self.evalID, originObj in tqdm(self.evalRes.items(),total=len(self.evalRes),desc="Processing"):
            totalCount+=1
            evalObj=copy.deepcopy(originObj)
            if evalObj['res']==1:
                print(f'question{self.evalID} is able to output right result, skip.')
                correctCount+=1
                continue
            if len(evalObj['solution'])>=maxRefineloop:
                print(f'question{self.evalID} has failed before,skip.')
                continue
            evalObj['solution']=[]
            print("processing question: ",self.evalID,' ',evalObj['question'])
            loop_count=0
            while evalObj['res']==0:
                print('round',loop_count+1)
                if len(evalObj['solution'])>1 and evalObj['solution'][-1]==evalObj['solution'][-2]:# if identical Reason was generated, probably fail.
                    print(f"identical result generated, giving up.")
                    break
                res=self.getSolution(evalObj)
                self.testSolution(evalObj)
                self.evaluate_res(evalObj)
                loop_count+=1
                if evalObj['res']==1:
                    print(f"{GREEN}Success!{RESET}corrected within {loop_count} loop")
                    correctCount+=1
                    print('ex: ',correctCount/totalCount,'\n----------------------------\n')
                if evalObj['res']==0 and loop_count>=maxRefineloop:
                    print(f"{RED}fail:{RESET}could not refine Reason of object in {maxRefineloop} loops, giving up.")
                    print('ex: ',correctCount/totalCount,'\n----------------------------\n')
                    break
            self.evalRes[self.evalID]=copy.deepcopy(evalObj)
            json.dump(self.evalRes,open(self.dump_path,'w'),indent=4)
            


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
            tool_choice="required",
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
        warnings.warn('tool did not call.')
        return {"error": "something went wrong, tool did not call."}


load_path='/home/walkiiiy/ChatTB/Process_document/V13/part_3.json'
dump_path='/home/walkiiiy/ChatTB/Process_document/V13/part_3.json'
db_root_path='/home/walkiiiy/ChatTB/Bird_dev/dev_databases'
getSolutionMCPserver_path='/home/walkiiiy/ChatTB/Process_document/V13/src/MCPserver_getSolution.py'
testMCPserver_path='/home/walkiiiy/ChatTB/Process_document/V13/src/MCPserver_testSql.py'

Processer=RuleProcesser(load_path=load_path,
                            dump_path=dump_path,
                            db_root_path=db_root_path,
                            solutionMCPserver_path=getSolutionMCPserver_path,
                            testMCPserver_path=testMCPserver_path,
                            )

# Processer.solution_loop(maxRefineloop=2)
# print(Processer.generate_schema_prompt(os.path.join(Processer.db_root_path, Processer.evalRes['0']['db_id'], Processer.evalRes['0']['db_id'] + '.sqlite')))

