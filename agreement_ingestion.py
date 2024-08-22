import re
import os
import sys
import json

from typing import List
from utils import pdf_to_nodes
from llama_index.core.schema import TextNode
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def agreements_ingestion(
    source_folder_path: str, 
    business_requests: List[str], 
    reader: PyMuPDFReader
) -> List[TextNode]:
    """
    This function extracts the agreement approvals for a list of business requests
    and stores them in a list of TextNodes
    
    Args:
        source_folder_path: path to the directory of business requests
        business_requests: a list of business request numbers indicating which requests to parse
        reader: a tool used to read PDFs
        
    """
    print("---------- Processing Agreement Approvals ----------")
    agreements = []
    BRs_with_agreements = []

    reg_compile = re.compile(".*Sign Off.*")
    for business_request in business_requests:
        business_request_path = f"{source_folder_path}/{business_request}"

        try:
            agreement_folder_path = None

            for dirpath, _, _ in os.walk(business_request_path):
                if (reg_compile.match(dirpath)):
                    agreement_folder_path = dirpath

            if agreement_folder_path == None:
                print(f"No signed agreement folder in {business_request}")
            else:                    
                pdf_files = [file for file in os.listdir(agreement_folder_path) 
                             if (file.endswith(".pdf"))]

                if (len(pdf_files) > 0):
                    BRs_with_agreements.append(business_request)

                for file in pdf_files:
                    nodes = pdf_to_nodes(
                        source_file_path=f"{agreement_folder_path}/{file}", 
                        business_request=business_request, 
                        file_category="Agreements", 
                        reader=reader
                    )
                    agreements += nodes

        except FileNotFoundError:
            print(f"File does not exist in {business_request}")
    
    print(f"Here are the business requests with agreement approvals: {BRs_with_agreements}")
    return agreements
    

if __name__ == "__main__":
    file = open("./config.json")
    config = json.load(file)

    reader = PyMuPDFReader()
    source_folder_path = config['source_folder_path']
    business_requests = os.listdir(source_folder_path)
    agreements = agreements_ingestion(source_folder_path, business_requests, reader)

    embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    Settings.embed_model = embed_model

    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(collection_name=config['agreement_collection_name'], client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex(
        nodes=agreements, 
        storage_context=storage_context, 
        embed_model=embed_model, 
        show_progress=True
    )
    