from mcp.server import FastMCP
from dotenv import load_dotenv
import os
import json


load_dotenv()
GAODE_KEY = os.getenv("GAODE_KEY")
CITY_CODE_FILE = os.getenv("CITY_CODE_FILE")
GOOGLE_API = os.getenv("GOOGLE_API")
DOCUMENT_PATH=os.getenv("TRAIN_DOCUMENT")
app = FastMCP("RulesProcessor")

@app.tool()
def rules_receiver(amends:str):
    """
    take in on paragraph of rules extracted from the amends. 
    Parameters:
     - rules (str): the rules extracted from the amends. 
    """
    pass

if __name__=="__main__":
    app.run(transport='stdio')