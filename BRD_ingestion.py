import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os
import sys
import json

from math import isnan
from typing import List, Callable
from utils import pdf_to_nodes

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import PyMuPDFReader
from qdrant_client import QdrantClient

def keep_text_only(cell):
    if isinstance(cell, float) and isnan(cell):
        return None

    return cell if isinstance(cell, str) else str(cell)
    
def excel_to_table(file_dir: str, sheetnames: List[str], keep_text_only: Callable) -> List[List[List[str]]]:
    sheets = []
    for sheetname in sheetnames:
        df = pd.read_excel(file_dir, sheet_name=sheetname)
        cleaned_df = df.dropna(how='all')
        text_only_df = cleaned_df.map(keep_text_only)
        sheet = []
        for i in range(len(text_only_df)):
            row = []
            num_columns = len(text_only_df.columns)
                    
            for j in range(num_columns):
                if (text_only_df.iloc[i, j] == None):
                    continue
                
                elif (text_only_df.iloc[i, j] == 'Select…'):
                    text_only_df.iloc[i, j] = 'None'
                row.append(text_only_df.iloc[i, j])
        
            sheet.append(row)        
        sheets.append(sheet)
    return sheets

def table_to_nodes(
    file_dir: str, 
    sheetnames: List[str], 
    sheets: List[List[List[str]]], 
    BR: str
) -> List[TextNode]:
    
    nodes = []
    idx = 0
    file_name = file_dir[file_dir.rfind('/') + 1:]
    
    for sheet in sheets:
        sheet_to_dict = {index: row for index, row in enumerate(sheet)}
        refined_sheet = {}
            
        #This section converts the table into JSON format
        for index, row in sheet_to_dict.items():
            item = {}

            if len(row) == 0:
                continue
            elif len(row) == 1:
                header = row[0]
                item[header] = []
            elif len(row) == 2:
                header = row[0] + ' ' + row[1]
                item[header] = []
            else:
                header = row[0] + ' ' + row[1]
                item[header] = row[2:]
                            
            refined_sheet[index] = item
    
        metadata = {"category": "BRD", "BR": BR, "filetype": "Spreadsheet", "source": file_name, "sheetname": sheetnames[idx]}
        nodes.append(TextNode(text=json.dumps(refined_sheet), metadata=metadata))
        
        idx += 1
    return nodes

def BRD_ingestion(in_dir: str, directories: List[str], reader: PyMuPDFReader) -> List[TextNode]:
    print("---------- Processing BRDs ----------")
    word_vec = []
    for dir in directories:
        BRD_dir = f"{in_dir}/{dir}/BRD"
        try:
            pdf_files = [file for file in os.listdir(BRD_dir) 
                    if file.endswith(".pdf") #this assumes that all BRD files in pdf format has "BRD" in their filenames
                    ]
            
            if (len(pdf_files) == 0):
                print(f"{dir} is empty")
                continue

            #older BRDs exist in Excel formats
            excel_files = [file for file in os.listdir(BRD_dir) if file.endswith('.xlsx') or file.endswith('xlsm') or file.endswith('xlsb') \
                    or file.endswith('XLSX') or file.endswith('XLSM') or file.endswith('XLSB')]

            used_files = []
            
            for file in excel_files:
                filename = file[:file.rfind('.')]
            
                excel_file = pd.ExcelFile(f"{BRD_dir}/{file}")
                sheetnames = [sheet.title for sheet in excel_file.book.worksheets if sheet.sheet_state == "visible"]
                
                sheets = excel_to_table(file_dir=f"{BRD_dir}/{file}", sheetnames=sheetnames, keep_text_only=keep_text_only)
                nodes = table_to_nodes(file_dir=f"{BRD_dir}/{file}", sheetnames=sheetnames, sheets=sheets, BR=dir)

                word_vec += nodes
                used_files.append(filename)

            #Newer BRDs only exist in PDF forms
            new_BRDs = [file for file in pdf_files if file[:file.rfind('.')] not in used_files]
            for file in new_BRDs:
                nodes = pdf_to_nodes(file_dir=f"{BRD_dir}/{file}", BR=dir, category="BRD", reader=reader)
                word_vec += nodes
                
        except FileNotFoundError:
            print(f"File does not exist in {dir}")

    return word_vec

if __name__ == "__main__":
    reader = PyMuPDFReader()
    in_dir = sys.argv[1]
    directories = os.listdir(in_dir)
    BRDs = BRD_ingestion(in_dir, directories, reader)

    embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    Settings.embed_model = embed_model

    client = QdrantClient(host="localhost", port=6333)

    vector_store = QdrantVectorStore(collection_name=sys.argv[2], client=client)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=BRDs, storage_context=storage_context, embed_model=embed_model, show_progress=True)


