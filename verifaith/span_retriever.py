import numpy as np
import faiss
import nltk
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def split_into_sentences(docs: list[str]) -> list[dict]:
    all_sentences = []
    for doc_idx, doc in enumerate(docs):
        sentences = nltk.sent_tokenize(doc.strip())
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:
                all_sentences.append({
                    "sentence": sent,
                    "doc_index": doc_idx
                })
    return all_sentences


def build_faiss_index(sentences: list[dict]):
    texts = [s["sentence"] for s in sentences]
    embeddings = model.encode(texts)
    embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    return index, sentences

def retrieve_span(claim: str, index, sentence_store: list[dict], top_k: int = 3) -> dict:
    
    claim_embedding = model.encode([claim])
    claim_embedding = claim_embedding / np.linalg.norm(
        claim_embedding, axis=1, keepdims=True
    )
    
    scores, indices = index.search(claim_embedding.astype('float32'), top_k)
    
    best_idx = indices[0][0]
    best_score = float(scores[0][0])
    best_sentence = sentence_store[best_idx]
    
    return {
        "sentence": best_sentence["sentence"],
        "doc_index": best_sentence["doc_index"],
        "similarity_score": round(best_score, 4),
        "low_confidence": best_score < 0.70,
        "top_3_spans": [
            {
                "sentence": sentence_store[indices[0][i]]["sentence"],
                "doc_index": sentence_store[indices[0][i]]["doc_index"],
                "score": round(float(scores[0][i]), 4)
            }
            for i in range(top_k)
        ]
    }