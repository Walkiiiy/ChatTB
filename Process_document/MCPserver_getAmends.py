from mcp.server import FastMCP
from dotenv import load_dotenv
import os
import json


load_dotenv()
GAODE_KEY = os.getenv("GAODE_KEY")
CITY_CODE_FILE = os.getenv("CITY_CODE_FILE")
GOOGLE_API = os.getenv("GOOGLE_API")
DOCUMENT_PATH=os.getenv("TRAIN_DOCUMENT")
app = FastMCP("AmendsProcessor")

@app.tool()
def amends_receiver(amends:str):
    """
    take in detailed text amends of the given worng sql. 
    Parameters:
     - amends (str): the amends of the wrong sql. 
    """
    pass

if __name__=="__main__":
    app.run(transport='stdio')