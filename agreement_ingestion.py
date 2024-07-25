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

def agreements_ingestion(in_dir: str, directories: List[str], reader: PyMuPDFReader) -> List[List[TextNode]]:
    print("---------- Processing Agreement Approvals ----------")
    agreements = []
    BRs_with_agreements = []

    reg_compile = re.compile(".*Sign Off.*")
    for directory in directories:
        full_dir = f"{in_dir}/{directory}"   
        try:
            results = []
            for dirpath, dirnames, filenames in os.walk(full_dir):
                if (reg_compile.match(dirpath)):
                    results.append(dirpath)

            if len(results) == 0:
                print(f"No sign off agreements in {directory}")
                continue

            # for now assume that there is only one sign off folder in each BR if there is one
            sign_off_dir = results[0]
                
            files = [file for file in os.listdir(sign_off_dir) 
                    if (file.endswith(".pdf")) 
                    ]
            
            if len(files) == 0:
                print(f"No sign off agreements in {directory}")
                continue
            else:
                BRs_with_agreements.append(directory)

            for file in files:
                nodes = pdf_to_nodes(file_dir=f"{sign_off_dir}/{file}", BR=directory, category="Agreements", reader=reader)
                agreements += nodes

        
        except FileNotFoundError:
            print(f"File does not exist in {directory}")
    
    print(f"BRs with Agreements: {BRs_with_agreements}")
    return agreements
    

if __name__ == "__main__":
    reader = PyMuPDFReader()
    in_dir = sys.argv[1]
    directories = os.listdir(in_dir)
    agreements = agreements_ingestion(in_dir, directories, reader)

    embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    Settings.embed_model = embed_model

    client = QdrantClient(host="localhost", port=6333)

    vector_store = QdrantVectorStore(collection_name=sys.argv[2], client=client)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=agreements, storage_context=storage_context, embed_model=embed_model, show_progress=True)