from Process_model.LLMClient import LLMClient
from Process_model.SchemaInformation import SchemaInformation
import os
import json

class weaken_NLQ:
    def __init__(self, dataset_path: str,output_path: str,model_path: str,db_root_path: str=None):
        self.client = LLMClient(model_path=model_path)  
        self.db_root_path = db_root_path
        self.schema_information = SchemaInformation()
        self.dataset_path = dataset_path
        self.output_path = output_path
    def weaken_NLQ(self, NLQ: str,db_path: str):
        prompt = f"""
        You are a helpful assistant that weakens the natural language question to align with normal user's inputs.
        You will be given a database schema and a natural language question around the schema.
        Your task is to weaken the question to align with normal user's inputs.
        You should find some table and column names in the question, replace them with different name with same meanings.
        Moreover, if you think the question is over specified, you should weaken the question to make it more general.
        HARD CONSTRAINTS:
        You should not change the meaning of the question.
        The weakened question should have exact same answer with the original question.
        You should only return the weakened question, DO NOT include any other text.

        The schema is:
        {self.schema_information.generate_schema_info(db_path,1)}
        The original question is:
        {NLQ}
        The weakened question is:
        """
        return self.client.chat(prompt)

    def main_loop(self):
        with open(self.dataset_path,'r') as f:
            data = json.load(f)
        for i,item in enumerate(data):
            # print(item)
            NLQ = data[item]['question']
            db_id = data[item]['db_id']
            db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
            weakened_NLQ = self.weaken_NLQ(NLQ,db_path)
            data[item]['weakened_question'] = weakened_NLQ
            print(f"processing question{i}:")
            print(f"original question: {NLQ}")
            print(f"weakened question: {weakened_NLQ}")
            print("--------------------------------")
            if i%20==0:
                with open(self.output_path,'w') as f:
                    json.dump(data,f,indent=4)
        with open(self.output_path,'w') as f:
            json.dump(data,f,indent=4)


if __name__ == "__main__":

    weaken_NLQ0 = weaken_NLQ(
        dataset_path='/home/ubuntu/walkiiiy/ChatTB/Bird_train/condensed_rules.json',
        output_path='/home/ubuntu/walkiiiy/ChatTB/Bird_train/weakened_dataset.json',
        model_path='/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B',
        db_root_path='/home/ubuntu/walkiiiy/ChatTB/Bird_train/database')
    weaken_NLQ0.main_loop()
    del weaken_NLQ0

    weaken_NLQ1 = weaken_NLQ(
        dataset_path='/home/ubuntu/walkiiiy/ChatTB/Bird_dev/condensed_rules.json',
        output_path='/home/ubuntu/walkiiiy/ChatTB/Bird_dev/weakened_dataset.json',
        model_path='/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B',
        db_root_path='/home/ubuntu/walkiiiy/ChatTB/Bird_dev/database')
    weaken_NLQ1.main_loop()
    