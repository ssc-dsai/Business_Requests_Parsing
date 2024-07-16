import pandas as pd
import os
import re
import sys
import pdfplumber

from typing import List, Dict, Any, Callable
from model import model
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader

lable_model = model.Model()
load_dotenv()

# Keeps only cell that contains texts
def keep_text_only(cell: Any) -> Any:
    return cell if isinstance(cell, str) else None

# Creates a label for each image found in a page
def extract_image_labels(cur_page: pdfplumber.page.Page, image_dir: str) -> List[Dict[str, Any]]:
    images = cur_page.images
    objs = []
    for idx, image in enumerate(images):
        #exclude very small images(random noises)
        if (abs(image['y1'] - image['y0']) < 1):
            continue
        bbox = [image['x0'], cur_page.cropbox[3]-image['y1'],  image['x1'], cur_page.cropbox[3]-image['y0']]
        img_page = cur_page.crop(bbox=bbox)
        img_obj = img_page.to_image(resolution=500)
        page_number = image['page_number']
        image_name_prefix = f'{page_number}-{idx + 1}'
        image_name = f'{image_name_prefix}' + ".png"
        image_path = f'{image_dir}/{image_name}'
        img_obj.save(image_path)

        obj = ({"label": lable_model.classify(image_path), "y0": image["y0"], "y1": image["y1"]})
        objs.append(obj)

    return objs


def get_label_mappings(file_dir: str, BR: str) -> Dict[str, str]:
    with pdfplumber.open(file_dir) as pdf:
        if ("Business Requirements Document" not in pdf.pages[0].extract_text_simple()):
            print(f"{file_dir} in {BR} is not a BRD")
            return {}
                
        mappings = {}
        for page in pdf.pages:
            labels = extract_image_labels(page, "./images/garbage")
            lines = page.extract_text_lines()
                    
            for label in labels:
                # This finds which row the lable is associated with
                for line in lines:
                    if (abs(label['y0'] - line['chars'][0]['y0']) <= 2 and abs(label['y1'] - line['chars'][0]['y1']) <= 2):
                        mappings[line['text']] = str(label['label'])
                        break
        return mappings

# This function cleans and reformats the table(extracted from the Excel file)
def excel_to_table(file_dir: str, sheetnames: List[str], keep_text_only: Callable) -> List[List[List[str]]]:
    sheets = []
    for sheetname in sheetnames:
        df = pd.read_excel(file_dir, sheet_name=sheetname)
        text_only_df = df.map(keep_text_only)
        cleaned_df = text_only_df.dropna(how='all')
        sheet = []
        for i in range(len(cleaned_df)):
            row = []
            num_columns = len(cleaned_df.columns)
                    
            for j in range(num_columns):
                if (cleaned_df.iloc[i, j] == None):
                    continue
                
                elif (cleaned_df.iloc[i, j] == 'Select…'):
                    cleaned_df.iloc[i, j] = 'None'
                row.append(cleaned_df.iloc[i, j])
        
            sheet.append(row)        
        sheets.append(sheet)
    return sheets

def table_to_nodes(
    file_dir: str, 
    sheetnames: List[str], 
    mapping: Dict[str, str], 
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
                        
            if len(row) >= 2:
                header_to_compare = (header[:header.index('\n')] if '\n' in header else header).rstrip()
                for key in mapping.keys():
                    if header_to_compare in key:
                        item[header].append(mapping[key])
                        break
                            
            refined_sheet[index] = item
    
        metadata = {"category": "BRD", "BR": BR, "filetype": "Spreadsheet", "source": file_name, "sheetname": sheetnames[idx]}
        nodes.append(TextNode(text=str(refined_sheet), metadata=metadata))
        
        idx += 1
    return nodes

def pdf_to_nodes(file_dir: str, BR: str, parser: LlamaParse) -> List[TextNode]:
    nodes = []
    pages = parser.load_data(file_dir)
    file_name = file_dir[file_dir.rfind('/') + 1:]

    #This makes sure that the pdf is a BRD
    if len(pages) >= 1:
        if "Business Requirements Document" not in pages[0].text:
            return nodes
    
    for num, page in enumerate(pages):
        rows = page.text.split('\n')

        #This keeps rows that actually contain content
        row_pattern = re.compile(r'[A-Za-z0-9]')
        rows = [row for row in rows if row_pattern.search(row)]
        
        json = {}
        for index, row in enumerate(rows):
            json[index] = row    

        metadata = {"category": "BRD", "BR": BR, "filetype": "PDF", "source": file_name, "page_number": (num + 1)}
        node = TextNode(text=str(json), metadata=metadata)
        nodes.append(node)
    return nodes

def BRD_ingestion(directories: List[str], parser: LlamaParse) -> List[TextNode]:
    print("---------- Processing BRDs ----------")
    word_vec = []

    #Iterating through all the BR directories
    for dir in directories:
        BRD_dir = f"{in_dir}/{dir}/BRD"
        try:
            pdf_files = [file for file in os.listdir(BRD_dir) 
                    if file.endswith(".pdf")
                    ]
            
            #Empty Directory
            if (len(pdf_files) == 0):
                print(f"{dir} is empty")
                continue

            #older BRDs exist both in excel and pdf forms
            #pdf_mappings = {pdf_name: mappings}
            pdf_mappings = {}
            for file in pdf_files:

                #For each pdf file, extract all the labels in that file
                #See Section 3.0 "Reason for Request" in BRDs that exist as Excel files
                mappings = get_label_mappings(f"{BRD_dir}/{file}", BR=dir)
                filename = file[:file.rfind('.')]
                pdf_mappings[filename] = mappings
                    
            excel_files = [file for file in os.listdir(BRD_dir) if file.endswith('.xlsx') or file.endswith('xlsm') or file.endswith('xlsb') \
                    or file.endswith('XLSX') or file.endswith('XLSM') or file.endswith('XLSB')]

            used_files = []
            
            for file in excel_files:
                filename = file[:file.rfind('.')]

                #This ensures that we are only looking at BRDs
                if filename not in pdf_mappings.keys():
                    continue
            
                excel_file = pd.ExcelFile(f"{BRD_dir}/{file}")
                sheetnames = [sheet.title for sheet in excel_file.book.worksheets if sheet.sheet_state == "visible"]
                mapping = pdf_mappings[filename]
                
                sheets = excel_to_table(file_dir=f"{BRD_dir}/{file}", sheetnames=sheetnames, keep_text_only=keep_text_only)
                nodes = table_to_nodes(file_dir=f"{BRD_dir}/{file}", sheetnames=sheetnames, mapping=mapping, sheets=sheets, BR=dir)

                word_vec += nodes
                used_files.append(filename)

            #Newer BRDs only exist in PDF forms
            new_BRDs = [file for file in pdf_files if file[:file.rfind('.')] not in used_files]
            for file in new_BRDs:
                nodes = pdf_to_nodes(f"{BRD_dir}/{file}", dir, parser)
                word_vec += nodes
            
        except FileNotFoundError:
            print(f"File does not exist in {dir}")

    return word_vec

def agreements_ingestion(directories: List[str]) -> List[List[TextNode]]:
    print("---------- Processing Agreement Approvals ----------")
    partitioned_agreements = []
    reg_compile = re.compile(".*Sign Off.*", flags=re.IGNORECASE)
    for dir in directories:
        full_dir = f"{in_dir}/{dir}"   
        try:
            results = []
            for dirpath, _, _ in os.walk(full_dir):
                if (reg_compile.match(dirpath, re.IGNORECASE)):
                    results.append(dirpath)

            if len(results) == 0:
                print(f"No sign off agreements in {dir}")
                continue

            # for now assume that there is only one sign off folder in each BR if there is one
            sign_off_dir = results[0]
                
            files = [file for file in os.listdir(sign_off_dir) 
                    if (file.endswith(".pdf")) 
                    ]
            
            if len(files) == 0:
                print(f"No sign off agreements in {dir}")
                continue

            agreements = []
            #Assume everything in sign agreement folder are agreement approvals
            for file in files:
                print(file)
                loader = PyPDFLoader(f"{sign_off_dir}/{file}", extract_images=True)
                elements = loader.load_and_split()

                for element in elements:
                    element.metadata['category']= 'Agreement Approval'
                    element.metadata['BR'] = dir
                    element.metadata['filetype'] = "PDF"
                    element.metadata['source'] = file

                    #offset the page number by 1
                    element.metadata['page'] += 1

                nodes = [TextNode(text=element.page_content, metadata=element.metadata) for element in elements]
                agreements.append(nodes)
                
            if (len(agreements) != 0):
                partitioned_agreements.append(agreements)
        except FileNotFoundError:
            print(f"File does not exist in {dir}")
    
    return partitioned_agreements

def chunk_data(nodes: List[List[TextNode]]) -> List[TextNode]:
    print("---------- Chunking Agreements ----------")
    embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )

    chunks = []
    for BR in nodes:
        for files in BR:
            semantic_chunkings = splitter._parse_nodes(files, show_progress=True)
            chunks = chunks + semantic_chunkings
            
    for chunk in chunks:
        for key in chunk.relationships.keys():
            chunk.metadata = chunk.relationships[key].metadata
    
    return chunks

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    parser = LlamaParse(
        #Obtained from https://cloud.llamaindex.ai/
        #https://github.com/run-llama/llama_parse
        api_key=os.getenv("LLAMA_PARSE_API_KEY"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=1,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

    in_dir = sys.argv[1]
    directories = os.listdir(in_dir)
    BRDs = BRD_ingestion(directories, parser)
    nodes = agreements_ingestion(directories)
    agreements = chunk_data(nodes)
    
    data = BRDs + agreements

    embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    Settings.embed_model = embed_model

    client = QdrantClient(host="localhost", port=6333)

    vector_store = QdrantVectorStore(collection_name=sys.argv[2], client=client)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=data, storage_context=storage_context, embed_model=embed_model, show_progress=True)