import re
import json

from typing import List

from llama_index.core.schema import TextNode
from llama_index.readers.file import PyMuPDFReader

def pdf_to_nodes(file_dir: str, BR: str, category: str, reader: PyMuPDFReader) -> List[TextNode]:
    pages = reader.load(file_dir)
    file_name = file_dir[file_dir.rfind('/') + 1:]
    valid_pattern = re.compile(r"(?i)(agreement|business requirements document)") #or BRD
    nodes = []
    if len(pages) >= 1:
        if valid_pattern.search(pages[0].text):
            for num, page in enumerate(pages):
                metadata = {"category": category, "BR": BR, "filetype": "PDF", "source": file_name, "page_number": (num + 1)}
                node = TextNode(text=page.text, metadata=metadata)
                nodes.append(node)

    return nodes