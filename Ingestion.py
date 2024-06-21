import os
import re
from typing import List
from model.model import classify

from unstructured.cleaners.core import clean

from llama_index.core.schema import TextNode
from llama_index.core.schema import Node
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

from langchain_community.document_loaders import PyPDFLoader
import pdfplumber

def parse_BRDs(in_dir: str) -> List[Node]:
    word_vec = []
    directories = os.listdir(in_dir)

    for dir in directories:
        BRD_dir = f"{in_dir}/{dir}/BRD"
        try:
            files = [file for file in os.listdir(BRD_dir) 
                    if file.endswith(".pdf") #this assumes that all BRD files in pdf format has "BRD" in their filenames
                    ]
            
            if (len(files) == 0):
                print(f"{dir} is empty")
                continue
                
            for file in files:
                with pdfplumber.open(f"{BRD_dir}/{file}") as pdf:

                    if ("Business Requirements Document" not in pdf.pages[0].extract_text_simple()):
                        print(f"{file} in {dir} is not a BRD")
                        continue

                    count = 1
                    
                    for page in pdf.pages:
                        labels = extract_image_labels(page, "./images/garbage")
                        lines = page.extract_text_lines()
                        text = ""
                        
                        for label in labels:
                            for line in lines:
                                if (abs(label['y0'] - line['chars'][0]['y0']) <= 2 and abs(label['y1'] - line['chars'][0]['y1']) <= 2):
                                    line['text'] += (': ' + str(label['label']))
                                    break
                
                        for line in lines:
                            text += (line['text'] + "\n")
                        
                        text = (clean(text, dashes=True))
                        metadata = {"category": "BRD", "BR": dir, "filetype": "PDF", "source": f"{BRD_dir}/{file}", "page_number": count}
                        word_vec.append(TextNode(text=text, metadata=metadata))

                        count += 1

        except FileNotFoundError:
            print(f"File does not exist in {dir}")
    
    return word_vec

def parse_agreements(in_dir: str) -> List[Node]:
    partitioned_agreements = []
    reg_compile = re.compile(".*Sign Off.*")
    directories = os.listdir(in_dir)

    for dir in directories:
        full_dir = f"{in_dir}/{dir}"   
        try:
            results = []
            for dirpath, _, _ in os.walk(full_dir):
                if (reg_compile.match(dirpath)):
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
                loader = PyPDFLoader(f"{sign_off_dir}/{file}", extract_images=True)
                elements = loader.load_and_split()

                for element in elements:
                    element.metadata['category']= 'Agreement Approval'
                    element.metadata['BR'] = dir
                    element.metadata['filetype'] = "PDF"

                nodes = [TextNode(text=element.page_content, metadata=element.metadata) for element in elements]
                agreements.append(nodes)
                
            if (len(agreements) != 0):
                partitioned_agreements.append(agreements)
            
        except FileNotFoundError:
            print(f"File does not exist in {dir}")
    
    return partitioned_agreements
        

def semantic_chunking_agreements(partitioned_agreements: List[Node]) -> List[Node]:
    embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )

    chunks = []
    for BR in partitioned_agreements:
        for files in BR:
            semantic_chunkings = splitter._parse_nodes(files, show_progress=True)
            chunks = chunks + semantic_chunkings
            
    for chunk in chunks:
        for key in chunk.relationships.keys():
            chunk.metadata = chunk.relationships[key].metadata
    
    return chunks


def extract_image_labels(cur_page, image_dir: str) -> List[dict]:
    images = cur_page.images
    objs = []
    for idx, image in enumerate(images):
        #don't include very small images(random noises)
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

        obj = ({"label": classify(image_path), "y0": image["y0"], "y1": image["y1"]})
        objs.append(obj)

    return objs

##after loading data, remember to add to qdrant
if __name__ == "__main__":
    in_dir = ""
    word_vec = parse_BRDs(in_dir)
    partitioned_agreements = parse_agreements(in_dir)
    chunks = semantic_chunking_agreements(partitioned_agreements)
    
    data = word_vec + chunks