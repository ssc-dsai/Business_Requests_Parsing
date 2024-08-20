import re

from typing import List

from llama_index.core.schema import TextNode
from llama_index.readers.file import PyMuPDFReader

def pdf_to_nodes(
    source_file_path: str, 
    business_request: str, 
    file_category: str, 
    reader: PyMuPDFReader
) -> List[TextNode]:
    """
    This function extracts the information from a pdf and stores them in a list of TextNodes
    
    Args:
        source_folder_path: path to the directory of business requests
        business_requests: a list of business request numbers indicating which requests to parse
        reader: a tool used to read PDFs
        
    """

    file_pages = reader.load(source_file_path)
    file_name = source_file_path[source_file_path.rfind('/') + 1:]
    valid_pattern = re.compile(r"(?i)(agreement|business requirements document|BRD)")
    business_request_number = ''.join(business_request.split(' '))
    nodes = []
    
    if len(file_pages) >= 1:
        if valid_pattern.search(file_pages[0].text):
            for page_num, page in enumerate(file_pages):
                metadata = {
                    "category": file_category, 
                    "BR": business_request_number, 
                    "filetype": "PDF", 
                    "source": file_name, 
                    "page_number": (page_num + 1)
                }
                node = TextNode(text=page.text, metadata=metadata)
                nodes.append(node)

    return nodes