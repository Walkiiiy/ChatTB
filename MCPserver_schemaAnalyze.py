from mcp.server import FastMCP
from dotenv import load_dotenv
import os
import json


load_dotenv()
GAODE_KEY = os.getenv("GAODE_KEY")
CITY_CODE_FILE = os.getenv("CITY_CODE_FILE")
GOOGLE_API = os.getenv("GOOGLE_API")
DOCUMENT_PATH=os.getenv("DEV_DOCUMENT")
app = FastMCP("schemaAnalyzeResultPrecessor")

@app.tool()
def description_receiver(databaseName:str,description:str):
    """
    take in natural language schema description generated, store for further process. 
    Parameters:
     - databaseName (str):the database (schema) name
     - description (str): natural language schema description generated
    """
    with open(os.path.join(DOCUMENT_PATH,databaseName+'.md'),'w') as f:
        f.write(description)

@app.tool()
def selected_word_recceiver(databaseName:str,selected_words:list[dict]):
    """
    take in selected words needs to supplement their background knowledge, store for further process. 
    Parameters:
     - databaseName (str):the database (schema) name
     - selected_words (list[dict]):the selected words and reasons list in form:
        [{"selectedWord":"...","reason":"..."},......]
    """
    with open(os.path.join(DOCUMENT_PATH,'DocsToSearch',databaseName+'.json'),'w') as f:
        json.dump(selected_words,f,indent=4)
if __name__=="__main__":
    app.run(transport='stdio')