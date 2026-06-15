from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://localhost:9200")
INDEX = "faq"

# Mapping + Indexing 
def create_index():
    es.indices.create(
        index=INDEX,
        mappings={
            "properties": {
                "question": {"type": "text"},
                "answer": {"type": "text"},
                "section": {"type": "text"},
                "course": {"type": "keyword"},
                "answer_vector": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    )


def index_docs(docs):
    helpers.bulk(es, [
        {"_index": INDEX, "_source": doc}
        for doc in docs
    ])


# lexical search
def bm25_search(q):
    res = es.search(
        index=INDEX,
        query={
            "multi_match": {
                "query": q,
                "fields": ["question", "answer", "section"]
                # weighted fields: 
                # "fields": ["question^3", "answer^1", "section^0.5"]
                # , "type": "best_fields"/"most_fields"/"cross_fields"/"phrase"/"phrase_prefix"
            }
        }
    )
    return res["hits"]["hits"]


# semantic search through embeddings 
from openai import OpenAI
client = OpenAI()

def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding


def vector_search(q):
    q_vec = embed(q)

    return es.search(
        index=INDEX,
        knn={
            "field": "answer_vector",
            "query_vector": q_vec,
            "k": 5,
            "num_candidates": 50
        }
    )


# hybrid search 
def hybrid_search(q):
    return {
        "bm25": bm25_search(q),
        "vector": vector_search(q)
    }



# with chunking: 
def chunk_text(text, size=200):
    words = text.split()
    return [
        " ".join(words[i:i+size])
        for i in range(0, len(words), size)
    ]


def index_faq(doc):
    chunks = chunk_text(doc["answer"])

    for i, chunk in enumerate(chunks):
        es.index(
            index="faq",
            document={
                "doc_id": doc["id"],
                "chunk_id": i,
                "question": doc["question"],
                "text": chunk,
                "course": doc["course"]
            }
        )

