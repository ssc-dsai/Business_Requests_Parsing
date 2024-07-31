import re
import os
import sys

from typing import List
from utils import pdf_to_nodes
from llama_index.core.schema import TextNode
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def agreements_ingestion(source_folder_path: str, business_requests: List[str], reader: PyMuPDFReader) -> List[TextNode]:
    print("---------- Processing Agreement Approvals ----------")
    agreements = []
    BRs_with_agreements = []

    reg_compile = re.compile(".*Sign Off.*")
    for business_request in business_requests:
        full_path = f"{source_folder_path}/{business_request}"   
        try:
            agreement_folders = []
            for dirpath, _, _ in os.walk(full_path):
                if (reg_compile.match(dirpath)):
                    agreement_folders.append(dirpath)

            if len(agreement_folders) == 0:
                print(f"No signed agreement folder in {business_request}")
            else:
                # for now assume that there is only one agreement folder in each business request if there exists one
                agreement_directory = agreement_folders[0]
                    
                files_in_agreement = [file for file in os.listdir(agreement_directory) 
                                        if (file.endswith(".pdf")) 
                                     ]
                if (len(files_in_agreement) > 0):
                    BRs_with_agreements.append(business_request)

                for file in files_in_agreement:
                    nodes = pdf_to_nodes(
                        file_directory=f"{agreement_directory}/{file}", 
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
    reader = PyMuPDFReader()
    source_folder_path = sys.argv[1]
    business_requests = os.listdir(source_folder_path)
    agreements = agreements_ingestion(source_folder_path, business_requests, reader)

    embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    Settings.embed_model = embed_model

    client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(collection_name=sys.argv[2], client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex(
        nodes=agreements, 
        storage_context=storage_context, 
        embed_model=embed_model, 
        show_progress=True
    )
    