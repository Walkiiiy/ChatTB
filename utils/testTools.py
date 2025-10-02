from Generator.Tools import Tools

tools = Tools(table_schema_path="Bird_dev/schema.json",db_root_path="Bird_dev/database",model_path="Process_model/models--Qwen3-8B",adapter_path="Process_model/models--Assumer_Mixed/checkpoint-13000")

# print(tools.get_specific_columns_info("card_games", ["abc","seq"]))
# print(tools.get_schema("card_games", 1))
print(tools.get_NL_description("card_games"))
# print(tools.get_rules('''when filtering by "Academic Year" in the frpm table''', "california_schools"))