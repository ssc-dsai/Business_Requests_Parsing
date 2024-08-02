import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os
import sys
import json
import re

from dotenv import load_dotenv
from math import isnan
from typing import List, Tuple
from utils import pdf_to_nodes

from llama_index.core import VectorStoreIndex, Settings, StorageContext, SummaryIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import PyMuPDFReader
from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from pandas import DataFrame

def keep_text_only(cell):
    if isinstance(cell, float) and isnan(cell):
        return None

    return cell if isinstance(cell, str) else str(cell)

def extract_worksheet(worksheet_dataframe: DataFrame) -> Tuple[List[List[str]], bool]:
    num_rows = len(worksheet_dataframe)
    worksheet = []

    is_BRD = False
    BRD_pattern = re.compile(re.escape("Business Requirements Document"), re.IGNORECASE)

    for row_num in range(num_rows):
        row = []
            
        num_columns = len(worksheet_dataframe.columns)
                    
        for column_num in range(num_columns):
            if (worksheet_dataframe.iloc[row_num, column_num] == None):
                continue  
            elif (worksheet_dataframe.iloc[row_num, column_num] == 'Select…'):
                worksheet_dataframe.iloc[row_num, column_num] = 'None'

            if (bool(BRD_pattern.search(worksheet_dataframe.iloc[row_num, column_num]))):
                is_BRD = True

            row.append(worksheet_dataframe.iloc[row_num, column_num])
        
        worksheet.append(row)
    
    return worksheet, is_BRD
    
def excel_to_table(file_directory: str, sheet_names: List[str]) -> List[List[List[str]]]:
    worksheets = []
    is_BRD = False

    for sheet_name in sheet_names:
        uncleaned_data = pd.read_excel(file_directory, sheet_name=sheet_name)
        processed_data = uncleaned_data.dropna(how='all')
        text_only_data = processed_data.map(keep_text_only)
        worksheet, is_BRD = extract_worksheet(text_only_data)
        worksheets.append(worksheet)
        
    #Return only BRDs for now
    if (not is_BRD):
        return []
    
    return worksheets

def worksheet_to_nodes(
    worksheet: List[List[str]], 
    worksheet_index: int, 
    business_request: str, 
    file_name: str, 
    sheet_names: List[str]
) -> List[TextNode]:
    
    nodes = []
    sheet_to_dict = {row_num: row for row_num, row in enumerate(worksheet)}
    cleaned_sheet = {}
    float_pattern = r'-?\d+\.\d+'
            
    #This section converts the table into JSON format
    for row_num, row in sheet_to_dict.items():
        row_content = {}

        if len(row) == 0:
            continue
        elif len(row) == 1:
            header = row[0]
            row_content[header] = []
        elif len(row) == 2:
            header = row[0] + ' ' + row[1]
            row_content[header] = []
        else:
            header = row[0] + ' ' + row[1]
            row_content[header] = row[2:]
            
        floats = re.findall(float_pattern, header)
        #Assume that the first number found in the header is the section number
        if len(floats) > 0:
            section = float(floats[0])
            #this divides the worksheet by sections
            if (section % 2 == 1.0 and section >= 3.0):
                metadata = {
                    "category": "BRD", 
                    "BR": business_request, 
                    "filetype": "Spreadsheet", 
                    "source": file_name, 
                    "sheetname": sheet_names[worksheet_index]
                }
                nodes.append(TextNode(text=json.dumps(cleaned_sheet), metadata=metadata))
                cleaned_sheet = {}

        cleaned_sheet[row_num] = row_content

    if (len(cleaned_sheet) > 0) :
        metadata = {
            "category": "BRD", 
            "BR": business_request, 
            "filetype": "Spreadsheet", 
            "source": file_name, 
            "sheetname": sheet_names[worksheet_index]
        }
        nodes.append(TextNode(text=json.dumps(cleaned_sheet), metadata=metadata))
    
    return nodes

def table_to_nodes(
    file_directory: str, 
    sheet_names: List[str], 
    worksheets: List[List[List[str]]], 
    business_request: str
) -> List[TextNode]:
    
    nodes = []
    worksheet_index = 0
    file_name = file_directory[file_directory.rfind('/') + 1:]
    
    for worksheet in worksheets:
        nodes += worksheet_to_nodes(
            worksheet=worksheet, 
            worksheet_index=worksheet_index, 
            business_request=business_request, 
            file_name=file_name, 
            sheet_names=sheet_names
        )

        worksheet_index += 1
        
    return nodes

def BRD_ingestion(source_folder_path: str, business_requests: List[str], reader: PyMuPDFReader) -> List[TextNode]:
    print("---------- Processing BRDs ----------")
    word_vec = []
    excel_pattern = r'\.(xlsx|xlsm|xlsb)$'
    pdf_pattern = r'\.(pdf)$'
    for business_request in business_requests:
        full_path = f"{source_folder_path}/{business_request}/BRD"
        try:
            pdf_files = [file for file in os.listdir(full_path) if re.search(pdf_pattern, file, re.IGNORECASE)]
            for file in pdf_files:
                nodes = pdf_to_nodes(
                    file_directory=f"{full_path}/{file}", 
                    business_request=business_request, 
                    file_category="BRD", 
                    reader=reader
                )
                word_vec += nodes

            excel_files = [file for file in os.listdir(full_path) if re.search(excel_pattern, file, re.IGNORECASE)] 
            for file in excel_files:
                nodes = excel_to_nodes(
                    full_path=full_path, 
                    file=file, 
                    business_request=business_request
                )
                word_vec += nodes


            #Newer BRDs only exist in PDF forms
            # new_BRDs = [file for file in pdf_files if file[:file.rfind('.')] not in used_files]
                
        except FileNotFoundError:
            print(f"File does not exist in {business_request}")

    return word_vec


def excel_to_nodes(full_path: str, file: str, business_request: str) -> List[TextNode]:
    excel_file = pd.ExcelFile(f"{full_path}/{file}")
    sheet_names = [sheet.title for sheet in excel_file.book.worksheets if sheet.sheet_state == "visible"]
                
    worksheets = excel_to_table(file_directory=f"{full_path}/{file}", sheet_names=sheet_names)

    nodes = table_to_nodes(
        file_directory=f"{full_path}/{file}", 
        sheet_names=sheet_names, 
        worksheets=worksheets, 
        business_request=business_request
    )

    return nodes


def generate_summaries(business_requests: List[str], BRDs: List[TextNode]) -> List[TextNode]:
    summaries = []
    for business_request in business_requests:
        nodes = [node for node in BRDs if node.metadata['BR'] == business_request]
        summary_index = SummaryIndex(nodes)
        query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize", 
            use_async=True, 
            llm=Settings.llm 
        )
        summary_response = query_engine.query("Please summarize this document and provide details about the request, key contacts, required services, project timelines, constraints, funding information, project management details, network and telecom requirements, security considerations, approval processes, service line details, location of the service required, and implementation activities.")
        node = TextNode(text=summary_response.response, metadata={"BR": business_request})
        summaries.append(node)
    
    return summaries

if __name__ == "__main__":
    load_dotenv()

    reader = PyMuPDFReader()
    source_folder_path = sys.argv[1]
    business_requests = os.listdir(source_folder_path)
    BRDs = BRD_ingestion(source_folder_path, business_requests, reader)

    embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    Settings.embed_model = embed_model
    Settings.llm = OpenAI(model="gpt-4o-mini", request_timeout=180, max_tokens=2048)

    client = QdrantClient(host="localhost", port=6333)

    business_request_summaries = generate_summaries(business_requests, BRDs)
    
    summaries_vector_store = QdrantVectorStore(
        collection_name=sys.argv[2],
        client=client,
    )

    details_vector_store = QdrantVectorStore(
        collection_name=sys.argv[3],
        client=client,
    )

    summaries_storage_context = StorageContext.from_defaults(vector_store=summaries_vector_store)
    details_storage_context = StorageContext.from_defaults(vector_store=details_vector_store)

    summaries_index = VectorStoreIndex(
        nodes=business_request_summaries, 
        storage_context=summaries_storage_context, 
        embed_model=embed_model, 
        show_progress=True
    )

    details_index = VectorStoreIndex(
        nodes=BRDs, 
        storage_context=details_storage_context, 
        embed_model=embed_model, 
        show_progress=True
    )

