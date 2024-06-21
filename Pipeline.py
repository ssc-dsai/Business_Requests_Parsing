import pandas as pd
import os
import qdrant_client
import phoenix as px
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import set_global_handler

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from llama_index.core.vector_stores import MetadataFilter
from llama_index.core.vector_stores import MetadataFilters

from llama_index.core import PromptTemplate

from Retriever import HybridRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank

px.launch_app()
set_global_handler("arize_phoenix")

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

#adjusting model for querying and embedding
embed_model = FastEmbedEmbedding("mixedbread-ai/mxbai-embed-large-v1")
# llm = Ollama(model="llama3", request_timeout=800, max_tokens=512)
llm = OpenAI(model="gpt-3.5-turbo", request_timeout=360, max_tokens=1024,  temperature=0.3)

Settings.embed_model = embed_model
Settings.llm = llm

#store the embeddings in a vector store
client = qdrant_client.QdrantClient(
    url="http://localhost:6333"
)

storage_name = "brd_agreements" #storage of the embeddings

vector_store = QdrantVectorStore(client=client, collection_name=storage_name, parallel=2)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model, show_progress=True)

def send_prompt(prompt, BR):
    # set up prompt template as well
    if BR is not None:
        filter = MetadataFilter(key="BR", value=BR)
        filters = MetadataFilters(filters=[filter])

        #fix top_k retrieval
        retriever = HybridRetriever(index.as_retriever(similarity_top_k=10, filters=filters), top_k=10)
        query_engine = RetrieverQueryEngine(
            retriever = retriever,
            node_postprocessors=[
               LLMRerank(
                    llm = llm
                )
            ]
        )
    else:
        retriever = HybridRetriever(index.as_retriever(similarity_top_k=10), top_k=10)
        query_engine = RetrieverQueryEngine(
            retriever = retriever,
            node_postprocessors=[
                LLMRerank(
                    llm = llm
                )
            ]
        )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": make_template()}
    )

    response = query_engine.query(prompt)
    return response

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