from typing import Union
import json
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from typing import Any


with open('documents-with-ids.json', 'rt') as f_in:
    documents = json.load(f_in)


model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)




for doc in tqdm(documents):
    question = doc['question']
    text = doc['text']
    qt = question + ' ' + text

    doc['question_vector'] = model.encode(question)
    doc['text_vector'] = model.encode(text)
    doc['question_text_vector'] = model.encode(qt)


es_client = Elasticsearch('http://localhost:9200') 

### Create index 
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)



from langchain.embeddings import SentenceTransformerEmbeddings
from typing import Dict
from langchain_elasticsearch import ElasticsearchRetriever

es_url = 'http://localhost:9200'

query = 'I just discovered the course. Can I still join it?'
course = "data-engineering-zoomcamp"

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

def make_elastic_retriever(
        content_field: str="text", 
        use_bm25: bool=True,
        use_vector: bool=True,
        bm25_boost: float=0.5,
        vector_boost: float=0.5,
        fields: Union[tuple[str],list[str]]=("question", "text", "section"),
        field_weights: Union[tuple[float],list[float]]=(1,1,1),  
        vector_field: Union[str, None]=None, 
        multi_match_type: str="best_fields",
        knn_k: int=5,
        num_candidates: int=10_000,
        use_rrf: bool=False,
        rank_constant: int=60, 
        size: int=5, 
        source: Union[list[str], None] = None, 
        metadata_filter: Union[dict[str, Any], None] = None, 
):
    def hybrid_query(search_query: str)-> Dict:
        if len(fields) != len(field_weights):
            weights = [1.0 for _ in fields]
        else:
            weights = field_weights
        weighted_fields = [f"{f}^{w}" for f,w in zip(fields, weights)] 

        use_bm25_local = use_bm25
        if not use_bm25_local and not use_vector: 
            use_bm25_local = True # activate one by default 
        
        use_rrf_local = (
            use_rrf 
            and use_bm25_local 
            and use_vector
        ) 

        if use_rrf_local and (bm25_boost != 1 or vector_boost != 1):
            print("Warning: boosts ignored with RRF")

        if use_vector and vector_field is None: 
            raise ValueError("Vector field unspecified.")

        body = {
            "size": size, 
        }

        if source is not None:
            body["_source"] = source

        # Text retrieval - BM25 
        if use_bm25_local: 
            body["query"] = {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": search_query, 
                            "fields": weighted_fields, 
                            "type": multi_match_type, 
                            "boost": bm25_boost, 
                        }
                    }
                }
            }

            if metadata_filter: 
                body["query"]["bool"]["filter"] = [
                    {
                        "term": {
                            k: v
                        }
                    }
                    for k,v in metadata_filter.items()
                ]

        # Vector search 
        if use_vector: 
            vector = embeddings.embed_query(search_query)  

            body["knn"] = {
                "field": vector_field, 
                "query_vector": vector, 
                "k": knn_k, 
                "num_candidates": num_candidates, 
                "boost": vector_boost, 
            }

            if metadata_filter: 
                body["knn"]["filter"] = [
                    {
                        "term": {
                            k: v
                        }
                    }
                    for k,v in metadata_filter.items()
                ]

        if use_rrf_local:
            body["rank"] = {"rrf": {"rank_constant": rank_constant}} 

        return body 
    
    return ElasticsearchRetriever.from_es_params(
        index_name=index_name,
        body_func=hybrid_query,
        content_field=content_field, 
        url=es_url,
    )


def retrieve(
        retriever: ElasticsearchRetriever, 
        search_query: str
    ) -> list[dict[str, Any]]: 
    hybrid_results = retriever.invoke(search_query)

    result_docs = []

    for hit in hybrid_results:
        result_docs.append({
            "content": hit.page_content,
            "metadata": hit.metadata
        })

    return result_docs 


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


def question_text_hybrid(q):
    question = q['question']
    course = q['course']

    return elastic_search_hybrid('question_text_vector', question, course)


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }



evaluate(ground_truth, question_text_hybrid)

