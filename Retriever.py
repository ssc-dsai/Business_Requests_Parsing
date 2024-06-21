from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever

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