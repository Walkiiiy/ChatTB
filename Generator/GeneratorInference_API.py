import openai
from Process_model.DeepSeekLLMClient import DeepSeekLLMClient
from Generator.Tools import Tools
from typing import Dict, Any, Optional
import re
import json
class Generator:
    def __init__(self, table_schema_path: str, model_path: str, db_root_path: str, adapter_path: str):
        self.client = DeepSeekLLMClient(api_key="api_key")
        self.tools = Tools(table_schema_path=table_schema_path, model_path=model_path, db_root_path=db_root_path, adapter_path=adapter_path)
    
    def generate_prompt(self, db_id: str, question: str,previous_content: str=None) -> str:
        pass
    
    def chat(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self.client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return response
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]):
        if tool_name == "get_rules":
            return self.tools.get_rules(**tool_args)
        elif tool_name == "get_specific_columns_info":
            return self.tools.get_specific_columns_info(**tool_args)
        elif tool_name == "get_schema":
            return self.tools.get_schema(**tool_args)
        elif tool_name == "get_NL_description":
            return self.tools.get_NL_description(**tool_args)
        else:
            print("Invalid tool name: {tool_name}")
            return f"Invalid tool name: {tool_name}, you should consider calling the right tool"
    
    def extract_tool_call(text, start_token="<CALL_TOOL>", end_token="</CALL_TOOL>"):
        """
        从模型返回的字符串中提取特殊 token 包裹的参数内容。
        支持 JSON 格式解析。
        
        参数:
            text (str): 模型输出的完整字符串
            start_token (str): 起始特殊 token
            end_token (str): 结束特殊 token
        
        返回:
            (tool_calls, errors)
            tool_calls: list[dict] -> 提取到的工具调用参数(JSON解析成功时)
            errors: list[str] -> JSON解析失败时的原始参数字符串
        """
        pattern = re.escape(start_token) + r"(.*?)" + re.escape(end_token)
        matches = re.findall(pattern, text, re.DOTALL)

        tool_calls = []
        errors = []

        for match in matches:
            raw_content = match.strip()
            try:
                parsed = json.loads(raw_content)
                tool_calls.append(parsed)
            except json.JSONDecodeError:
                errors.append(raw_content)

        return tool_calls, errors
    
    def NLQ_processing(question: str,db_id: str)->str:
        prompt=generate_prompt(db_id,question)
        response=self.chat(**prompt)
        tool_calls, errors=self.extract_tool_call(response)
        while tool_calls:
            pass

    def Plan_generate(self,db_id: str,question: str,previous_content: str=None)->str:
        pass

    def SQL_generate(self,db_id: str,question: str,previous_content: str=None)->str:
        pass
    
    def Data_iterator():
        pass

    def main_loop(self):
        pass
