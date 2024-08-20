import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os
import sys
import json
import re

from dotenv import load_dotenv
from math import isnan
from typing import List, Tuple, Any
from utils import pdf_to_nodes

from llama_index.core import VectorStoreIndex, Settings, StorageContext, SummaryIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import PyMuPDFReader
from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from pandas import DataFrame

def BRD_ingestion(
    source_folder_path: str, 
    business_requests: List[str], 
    reader: PyMuPDFReader
) -> List[TextNode]:
    """
    This function converts a directory of business requests in a specific structure into a list of TextNodes, 
    which can then be interacted with LlamaIndex
    
    Args:
        source_folder_path: path to the directory of business requests
        business_requests: a list of business request numbers indicating which requests to parse
        reader: a tool used to read PDFs

    """
    print("---------- Processing BRDs ----------")

    business_request_nodes = []
    excel_pattern = r'\.(xlsx|xlsm|xlsb)$'
    pdf_pattern = r'\.(pdf)$'
    BRD_pattern = re.compile(".*BRD.*")

    for business_request in business_requests:
        try:
            business_request_path = f"{source_folder_path}/{business_request}"  
            BRD_folder_path = None
            for root, _, _ in os.walk(business_request_path):
                if (BRD_pattern.match(root)):
                    #Here assume that the first BRD like folder found contains the BRD
                    BRD_folder_path = root
                    break
                    
            if BRD_folder_path == None:
                print(f"No BRD folders in {business_request}")
            else:    
                pdf_files = [file for file in os.listdir(BRD_folder_path) if re.search(pdf_pattern, file, re.IGNORECASE)]
                
                for file in pdf_files:
                    nodes = pdf_to_nodes(
                        source_file_path=f"{BRD_folder_path}/{file}", 
                        business_request=business_request, 
                        file_category="BRD", 
                        reader=reader
                    )
                    business_request_nodes += nodes
    
                excel_files = [file for file in os.listdir(BRD_folder_path) if re.search(excel_pattern, file, re.IGNORECASE)] 
                for file in excel_files:
                    nodes = excel_to_nodes(
                        source_file_path=f"{BRD_folder_path}/{file}", 
                        business_request=business_request
                    )
                    business_request_nodes += nodes
                
        except FileNotFoundError:
            print(f"File does not exist in {business_request}")

    return business_request_nodes

def excel_to_nodes(
    source_file_path: str, 
    business_request: str
) -> List[TextNode]:
    """
    This function converts a BRD stored in an Excel file to a list of TextNodes
    
    Args:
        source_file_path: path to the Excel file
        business_request: the business request number of the interested BRD
        
    """
    excel_file = pd.ExcelFile(source_file_path)
    sheet_names = [sheet.title for sheet in excel_file.book.worksheets if sheet.sheet_state == "visible"]      
    worksheets = excel_to_table(source_file_path=source_file_path, sheet_names=sheet_names)

    nodes = table_to_nodes(
        source_file_path=source_file_path, 
        sheet_names=sheet_names, 
        worksheets=worksheets, 
        business_request=business_request
    )

    return nodes

def excel_to_table(
    source_file_path: str, 
    sheet_names: List[str]
) -> List[List[List[str]]]:
    """
    This function converts extracts the information of an Excel file by each worksheet
    and retains the information in a tabular structure using arrays
    
    Args:
        source_file_path: path to the Excel file
        sheet_names: a list of names for all the worksheets that are visible
        
    """
    worksheets = []
    is_BRD = False

    for sheet_name in sheet_names:
        uncleaned_data = pd.read_excel(source_file_path, sheet_name=sheet_name)
        processed_data = uncleaned_data.dropna(how='all')
        text_only_data = processed_data.map(keep_text_only)
        worksheet, valid_BRD = extract_worksheet(text_only_data)
        worksheets.append(worksheet)

        if valid_BRD:
            is_BRD = True
        
    #Return only BRDs for now
    if (not is_BRD):
        return []
    
    return worksheets

def keep_text_only(
    cell: Any
) -> str:
    """
    This function ensures that a dataframe entry is not nan, and if its not a string,
    then convert it to a string
    
    Args:
        cell: value of the dataframe entry
        
    """
    if isinstance(cell, float) and isnan(cell):
        return None

    return cell if isinstance(cell, str) else str(cell)

def extract_worksheet(
    worksheet_dataframe: DataFrame
) -> Tuple[List[List[str]], bool]:
    """
    This function ensures stores the information of a worksheet in a 2D array but also checks
    whether the worksheet is a BRD
    
    Args:
        worksheet_dataframe: the dataframe that stores the information of a worksheet
        
    """
    num_rows = len(worksheet_dataframe)
    worksheet = []

    valid_BRD = False
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
                valid_BRD = True

            row.append(worksheet_dataframe.iloc[row_num, column_num])
        
        worksheet.append(row)
    
    return worksheet, valid_BRD

def table_to_nodes(
    source_file_path: str, 
    sheet_names: List[str], 
    worksheets: List[List[List[str]]], 
    business_request: str
) -> List[TextNode]:
    """
    This function converts a BRD stored in an array structure into a list of TextNodes
    
    Args:
        source_file_path: path to the BRD file
        sheet_names: a list of names for all the worksheets in the BRD
        worksheets: a 3D array representing all the information in the BRD (worksheet, row, column)
        business_request: business request number of the BRD

    """
    
    nodes = []
    worksheet_index = 0
    file_name = source_file_path[source_file_path.rfind('/') + 1:]
    
    for worksheet in worksheets:
        nodes += worksheet_to_nodes(
            worksheet=worksheet, 
            worksheet_name=sheet_names[worksheet_index],
            business_request=business_request, 
            file_name=file_name, 
        )

        worksheet_index += 1
        
    return nodes

def worksheet_to_nodes(
    worksheet: List[List[str]], 
    worksheet_name: str, 
    business_request: str, 
    file_name: str, 
) -> List[TextNode]:
    """
    This function converts the information in a worksheet of a BRD into a list of TextNodes
    
    Args:
        worksheet: a 2D array representing all the information in the worksheet (row, column)
        worksheet_name: the name of the worksheet
        business_request: business request number of the BRD
        file_name: name of the file which is used as metadata of the nodes
        
    """
    
    nodes = []
    sheet_to_dict = {row_num: row for row_num, row in enumerate(worksheet)}
    cleaned_sheet = {}
    float_pattern = r'-?\d+\.\d+'

    business_request_number = ''.join(business_request.split(' '))
            
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
                    "BR": business_request_number, 
                    "filetype": "Spreadsheet", 
                    "source": file_name, 
                    "sheetname": worksheet_name
                }
                nodes.append(TextNode(text=json.dumps(cleaned_sheet), metadata=metadata))
                cleaned_sheet = {}

        cleaned_sheet[row_num] = row_content

    if (len(cleaned_sheet) > 0) :
        metadata = {
            "category": "BRD", 
            "BR": business_request_number, 
            "filetype": "Spreadsheet", 
            "source": file_name, 
            "sheetname": worksheet_name
        }
        nodes.append(TextNode(text=json.dumps(cleaned_sheet), metadata=metadata))
    
    return nodes

def generate_summaries(
    business_requests: List[str], 
    BRDs: List[TextNode]
) -> List[TextNode]:
    """
    This function takes in a list of business requests and generates a summary for each
    of the business requests using Generative AI
    
    Args:
        business_requests: a list of business requests numbers representing the business requests 
                           that need to be summarized
        BRDs: a list of TextNodes that stores the information of the BRDs
        
    """

    summaries = []
    for business_request in business_requests:
        business_request_number = ''.join(business_request.split(' '))
        nodes = [node for node in BRDs if node.metadata['BR'] == business_request_number]
        summary_index = SummaryIndex(nodes)
        query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize", 
            use_async=True, 
            llm=Settings.llm 
        )
        summary_response = query_engine.query("Please summarize this document and provide details about the request, key contacts, required services, project timelines, constraints, funding information, project management details, network and telecom requirements, security considerations, approval processes, service line details, location of the service required, and implementation activities.")
        node = TextNode(text=summary_response.response, metadata={"BR": business_request_number})
        summaries.append(node)
    
    return summaries

if __name__ == "__main__":
    load_dotenv()

    #Uncomment this section to see the backstage of the pipeline
    # import phoenix as px
    # px.launch_app()
    # from llama_index.core import set_global_handler
    # set_global_handler("arize_phoenix")

    reader = PyMuPDFReader()
    source_folder_path = sys.argv[1]
    business_requests = os.listdir(source_folder_path)
    BRDs = BRD_ingestion(source_folder_path, business_requests, reader)

    embed_model = FastEmbedEmbedding(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        max_length=1024,
        cache_dir="./embedding_cache"
    )
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

