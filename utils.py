import re

from typing import List

from llama_index.core.schema import TextNode
from llama_index.readers.file import PyMuPDFReader

def pdf_to_nodes(file_directory: str, business_request: str, file_category: str, reader: PyMuPDFReader) -> List[TextNode]:
    file_pages = reader.load(file_directory)
    file_name = file_directory[file_directory.rfind('/') + 1:]
    valid_pattern = re.compile(r"(?i)(agreement|business requirements document)") #or BRD
    nodes = []
    if len(file_pages) >= 1:
        if valid_pattern.search(file_pages[0].text):
            for page_num, page in enumerate(file_pages):
                metadata = {
                    "category": file_category, 
                    "BR": business_request, 
                    "filetype": "PDF", 
                    "source": file_name, 
                    "page_number": (page_num + 1)
                }
                node = TextNode(text=page.text, metadata=metadata)
                nodes.append(node)

    return nodes