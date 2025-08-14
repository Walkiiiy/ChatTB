from mcp.server import FastMCP
from dotenv import load_dotenv
import os
import json


load_dotenv()
GAODE_KEY = os.getenv("GAODE_KEY")
CITY_CODE_FILE = os.getenv("CITY_CODE_FILE")
GOOGLE_API = os.getenv("GOOGLE_API")
DOCUMENT_PATH=os.getenv("TRAIN_DOCUMENT")
app = FastMCP("Precessor")

@app.tool()
def reasoning_receiver(reason:str):
    """
    take in natural reasoning process to solve the question. 
    Parameters:
     - reason (str): the reasoning process
    """
    pass

@app.tool()
def SQLite_receiver(sql:str):
    """
    take in the final SQLite query you generate based on the question and the schema. 
    Parameters:
     - sql (str): the SQLite query
    """
    pass
if __name__=="__main__":
    app.run(transport='stdio')