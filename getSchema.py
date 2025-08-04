import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from io import StringIO
import json


load_dotenv()
DATABASE=os.getenv("TRAIN_DATABASE")
DESCRIPTION=os.getenv("TRAIN_DESCRIPTION")
MAX_SAMPLE_NUM=5

# return table_schema, list of column
def table_sampler(csvFile,descFile):
    print(f'processing {csvFile}')
    res={"tableName":csvFile.split('/')[-1]}

    # table=pd.read_csv(csvFile,encoding='latin1')
    # description=pd.read_csv(descFile,encoding='latin1')
    with open(descFile, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    description = pd.read_csv(StringIO(content))
    with open(csvFile, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    table = pd.read_csv(StringIO(content))

    colNum=0
    for col in table:
        col_normalized = col.strip().lower()
        description["original_column_name_normalized"] = description["original_column_name"].str.strip().str.lower()
        filtered_description = description[description["original_column_name_normalized"] == col_normalized]
        col_desc = column_sampler(table[col], filtered_description)
        res[f'column{colNum}'] = col_desc
        colNum += 1
    return res

# return column_description, dict
def column_sampler(column,descDF):
    col_desc={}

    if descDF.empty:
        return
    if 'original_column_name' in descDF and not pd.isna(descDF.iloc[0]['original_column_name']):
        col_desc['originColumnName'] = descDF.iloc[0]['original_column_name']
    if 'column_name' in descDF and not pd.isna(descDF.iloc[0]['column_name']):
        col_desc['fullColumnName'] = descDF.iloc[0]['column_name']
    if 'column_description' in descDF and not pd.isna(descDF.iloc[0]['column_description']):
        col_desc['columnDescription'] = descDF.iloc[0]['column_description']
    if 'data_format' in descDF and not pd.isna(descDF.iloc[0]['data_format']):
        col_desc['dataFormat'] = descDF.iloc[0]['data_format']
    if 'value_description' in descDF and not pd.isna(descDF.iloc[0]['value_description']):
        col_desc['valueDescription'] = descDF.iloc[0]['value_description']
    
    col_desc['size'] = column.size
    col_desc['emptyValueCount'] = int(column.isna().sum())

    non_null_column = column.dropna().reset_index(drop=True)

    solidNum = len(non_null_column)  # Solid numbers are non-null values
    unique_vals = set(non_null_column)
    

    # if str,calculate average example length.set example num in continuous columns
    adj_sample_num=MAX_SAMPLE_NUM
    if column.dtype not in [int,float,bool] and len(unique_vals)>0:
        totalsize=0
        for item in unique_vals:
            totalsize+=len(str(item))
        ave_size=totalsize/len(unique_vals)
        if ave_size>20:
            adj_sample_num=3
        if ave_size>50:
            adj_sample_num=2
        if ave_size>100:
            adj_sample_num=1
    adj_sample_num=min(MAX_SAMPLE_NUM,adj_sample_num)


    #determain continuous or discrete 
    S = len(unique_vals)
    L = solidNum
    if L==0:
        return col_desc
    MAX_SAMPLE_NUM
    if S/L>0.3:#continuous
        col_desc['valType']='continuous'

        if column.dtype in [int,float,bool]:
            # if continuous, take 5 random example
            col_desc['samples'] = [float(non_null_column.iloc[i]) for i in np.random.randint(0, solidNum, min(adj_sample_num, solidNum))]
            
            col_desc['averageValue'] = round(float(column.mean()), 3)
            col_desc['maximumValue'] = float(column.max())
            col_desc['minimumValue'] = float(column.min())
            col_desc['sampleVariance'] = round(float(column.var()), 2)
        else:
            col_desc['samples'] = [non_null_column.iloc[i] for i in np.random.randint(0, solidNum, min(adj_sample_num, solidNum))]
    else:# discrete
        col_desc['valType']='discrete'
        col_desc['typeNum']=S
        if column.dtype in [int,float,bool]:
            # if discrete,take 10 examples in order.
            col_desc['samples'] = [float(i) for _,i in enumerate(unique_vals)][:10]
            col_desc['averageValue'] = round(float(column.mean()), 3)
            col_desc['maximumValue'] = float(column.max())
            col_desc['minimumValue'] = float(column.min())
            col_desc['sampleVariance'] = round(float(column.var()), 2)
        else:
            col_desc['samples'] = [i for _,i in enumerate(unique_vals)][:10]
    return col_desc

# take in a single database route, return lists of table and description file routes
def getDatabaseCsvRoutes(databasePath):
    table_routes=[]
    desc_routes=[]
    for root,dirs,files in os.walk(databasePath):
        if root.split('/')[-1]=='database_description':#description dirs
           for file in files:
                if file.split('.')[-1]=='csv':
                    desc_routes.append(os.path.join(root,file))
        else:
            for file in files:
                if file.split('.')[-1]=='csv' and file!='sqlite_sequence.csv':# table csv
                    table_routes.append(os.path.join(root,file))
    return desc_routes,table_routes

if __name__=="__main__":
    for database in os.listdir(DATABASE):
        if database==".DS_Store" or database=="train_tables.json":
            continue
        # if os.path.exists(os.path.join(DESCRIPTION,f'{database}.json')):
        #     print(database,'already described.')
        #     continue
        databasePath=os.path.join(DATABASE,database)    
        desc_routes,table_routes=getDatabaseCsvRoutes(databasePath)
        databaseDesc={"databaseName":database}
        for i,_ in enumerate(table_routes):
            databaseDesc[f'table{i}']=table_sampler(table_routes[i],desc_routes[i])
        # print(databaseDesc)
        with open(os.path.join(DESCRIPTION,f'{database}.json'),'w') as f:
            json.dump(databaseDesc,f,indent=4)