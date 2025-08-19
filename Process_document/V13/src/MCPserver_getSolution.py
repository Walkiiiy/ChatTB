from mcp.server import FastMCP
from dotenv import load_dotenv
import os
import json


load_dotenv()
GAODE_KEY = os.getenv("GAODE_KEY")
CITY_CODE_FILE = os.getenv("CITY_CODE_FILE")
GOOGLE_API = os.getenv("GOOGLE_API")
DOCUMENT_PATH=os.getenv("TRAIN_DOCUMENT")
app = FastMCP("SolutionProcessor")

@app.tool()
def solution_receiver(solution:str):
    """
    take in detailed text solution of the given correct sql of the question. 
    Parameters:
     - solution (str): the solution of the question. 
    """
    pass

if __name__=="__main__":
    app.run(transport='stdio')