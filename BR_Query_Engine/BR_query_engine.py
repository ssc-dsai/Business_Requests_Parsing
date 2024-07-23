"""
title: Llama Index Pipeline
author: open-webui, Ian Yu
date: 2024-06-21
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
from llama_index.core import PromptTemplate, get_response_synthesizer
from llama_index.core.vector_stores import MetadataFilter
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from dotenv import load_dotenv

class Pipeline:
    def __init__(self):
        pass

    async def on_startup(self):
        import os

        # Set the OpenAI API key Here
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        from llama_index.core import VectorStoreIndex, Settings, set_global_handler
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        import qdrant_client

        client = qdrant_client.QdrantClient(
             host='host.docker.internal', port=6333
        )
 
        #adjusting model for querying and embedding
        embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
        llm = OpenAI(model=os.getenv("OPENAI_MODEL"), request_timeout=360, max_tokens=1024,  temperature=0.3)
        Settings.embed_model = embed_model
        Settings.llm = llm

        storage_name = os.getenv("COLLECTION_NAME") #storage of the embeddings
        vector_store = QdrantVectorStore(client=client, collection_name=storage_name, parallel=2)
        
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model, show_progress=True)
        self.llm = llm
        self.embed_model = embed_model
        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.
   # set up prompt template as well

        BR = None
        if "<" in user_message and ">" in user_message:
            start_idx = user_message.index("<")
            end_idx = user_message.index(">")
        else:
            start_idx = -1
            end_idx = -1

        if start_idx != -1 and end_idx != -1:
            BR = user_message[start_idx + 1:end_idx]
            user_message = user_message[end_idx + 1:]
            print(user_message)
    
        response_synthesizer = get_response_synthesizer(streaming=True)

        if BR is not None:
            filter = MetadataFilter(key="BR", value=BR)
            filters = MetadataFilters(filters=[filter])

            #fix top_k retrieval
            retriever = HybridRetriever(self.index.as_retriever(similarity_top_k=10, filters=filters), top_k=10)
            query_engine = RetrieverQueryEngine(
                retriever = retriever,
                response_synthesizer=response_synthesizer
            )
        else:
            retriever = HybridRetriever(self.index.as_retriever(similarity_top_k=10), top_k=10)
            query_engine = RetrieverQueryEngine(
                retriever = retriever,
                response_synthesizer=response_synthesizer
            )

        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": make_template()}
        )

        response = query_engine.query(user_message)
        return response.response_gen
    
def make_template():
    prompt = """\
            HUMAN
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. 
            Make sure you provide the source of your answer, which is the file that your information comes from.
            Context information is below.
            ---------------------
            {context_str}
            ---------------------
            Given the context information and not prior knowledge, answer the query.
            Query: {query_str}
            Answer: \
            """

    template = PromptTemplate(prompt)
    return template

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, top_k):
        self.vector_retriever = vector_retriever
        self.top_k = top_k
        super().__init__()

    #retrieves top 10 most similiar documents by default
    def _retrieve(self, query, **kwargs):      
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        nodes = [node.node for node in vector_nodes]
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10, **kwargs)
        bm25_nodes = bm25_retriever.retrieve(query, **kwargs)
        
        # combine the two lists of nodes
        score_nodes = { node.node.id_: node.score for node in vector_nodes}
        all_nodes = { node.node.id_: node for node in vector_nodes } 

        # uncomment to enable keyword search
        # for node in bm25_nodes:
        #     score_nodes[node.node.id_] += node.score * 0.1

        results = []

        sorted_nodes = dict(sorted(score_nodes.items(), key=lambda item: item[1], reverse=True))

        count = 0
        for node_id, score in sorted_nodes.items():
            if count == self.top_k:
                break

            all_nodes[node_id].score = score
            results.append(all_nodes[node_id])
            count += 1

        return results