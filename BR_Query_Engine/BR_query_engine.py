"""
title: Llama Index Pipeline
author: open-webui, Ian Yu
date: 2024-06-21
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

import qdrant_client
import os
import re

from typing import List, Union, Generator, Iterator
from llama_index.core import PromptTemplate, get_response_synthesizer, VectorStoreIndex, Settings, set_global_handler
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterCondition
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from dotenv import load_dotenv

class Pipeline:
    def __init__(self):
        pass

    async def on_startup(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        # Uncomment this section to see the backstage of the pipeline
        # import phoenix as px
        # px.launch_app()
        # from llama_index.core import set_global_handler
        # set_global_handler("arize_phoenix")
        # End of LLM Tracing

        #if running this pipeline using docker, then host should be host.docker.internal, otherwise it should be localhost
        client = qdrant_client.QdrantClient(
             host='host.docker.internal', port=6333
        )
 
        #this adjusts the large language model that will be used for querying and embedding
        embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
        llm = OpenAI(model=os.getenv("OPENAI_MODEL"), request_timeout=180, max_tokens=2048)

        Settings.embed_model = embed_model
        Settings.llm = llm

        self.embed_model = embed_model
        self.llm = llm

        BR_summaries_vector_store = QdrantVectorStore(
            collection_name=os.getenv("BR_SUMMARIES_COLLECTION_NAME"),
            client=client,
        )

        BR_details_vector_store = QdrantVectorStore(
            collection_name=os.getenv("BR_DETAILS_COLLECTION_NAME"),
            client=client,
        )

        self.BR_summaries_index = VectorStoreIndex.from_vector_store(
            BR_summaries_vector_store, 
            embed_model=embed_model, 
            show_progress=True
        )
        
        self.BR_details_index = VectorStoreIndex.from_vector_store(
            BR_details_vector_store, 
            embed_model=embed_model, 
            show_progress=True
        )

        pass

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        response_synthesizer = get_response_synthesizer(streaming=True)

        BR_pattern = r'BR\d+'
        BR_summaries_query_engine = self.BR_summaries_index.as_query_engine(similarity_top_k=10)

        summaries_response = BR_summaries_query_engine.query(user_message).response
        relevant_BRs = re.findall(BR_pattern, summaries_response + " " + user_message)
        
        filters = [MetadataFilter(key="BR", value=BR) for BR in relevant_BRs]
        filters = MetadataFilters(filters=filters, condition=FilterCondition.OR)
        
        retriever = HybridRetriever(self.BR_details_index.as_retriever(similarity_top_k=10, filters=filters), top_k=10)
        query_engine = RetrieverQueryEngine(
            retriever = retriever,
            response_synthesizer=response_synthesizer
        )

        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": make_template()}
        )

        response = query_engine.query(user_message + " " + summaries_response)
        return response.response_gen
    
def make_template() -> PromptTemplate:
    prompt = f"""\
            HUMAN
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. 
            List out all sources of your answer.
            Context information is below.
            ---------------------
            {{context_str}}
            ---------------------
            Given the context information and not prior knowledge, answer the query.
            Query: {{query_str}}
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