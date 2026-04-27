from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import json
import nltk
import numpy as np
import faiss

# Import models at startup
from verifaith.startup import groq_client, embed_model, nli_model

# ── APP SETUP ─────────────────────────────────────────────────

app = FastAPI(
    title="VeriFaith API",
    description="Verifiable Faithfulness Evaluation for RAG Systems",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────

class EvaluateRequest(BaseModel):
    answer: str = Field(
        ...,
        description="The RAG-generated answer to evaluate",
        example="The Eiffel Tower was built in 1889 and stands 330 metres tall."
    )
    source_docs: List[str] = Field(
        ...,
        description="List of source documents the answer was generated from",
        min_items=1
    )
    include_full_report: Optional[bool] = Field(
        default=True,
        description="Include per-claim breakdown in response"
    )

class ClaimResultResponse(BaseModel):
    claim_id:        int
    claim_text:      str
    status:          str
    confidence:      float
    evidence_span:   str
    source_doc:      int
    was_reranked:    bool

class EvaluateResponse(BaseModel):
    verifaith_version:   str
    evaluated_at:        str
    faithfulness_score:  float
    verdict:             str
    total_claims:        int
    supported:           int
    unverified:          int
    contradicted:        int
    has_contradictions:  bool
    claim_breakdown:     Optional[List[ClaimResultResponse]]
    contradiction_report: List[dict]
    warnings:            dict


# ── PIPELINE FUNCTIONS ────────────────────────────────────────

def extract_claims(answer: str) -> list[dict]:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a claim decomposition engine for AI evaluation.
Break down any given text into individual atomic factual claims.
Rules:
- Each claim must contain ONE fact only
- Each claim must be self-contained and understandable alone
- Each claim must be a declarative statement
- Do NOT include opinions or vague statements
- Do NOT merge two facts into one claim
Return ONLY a valid JSON object with key "claims" containing an array.
Format: {"claims": [{"claim_id": 1, "text": "..."}, {"claim_id": 2, "text": "..."}]}"""
            },
            {
                "role": "user",
                "content": f"Decompose this answer into atomic claims:\n\n{answer}"
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    raw    = response.choices[0].message.content.strip()
    raw    = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    
    # ── Robust parsing — handle all response formats ──────────
    
    # Format 1: {"claims": [...]}  ← what we asked for
    if "claims" in parsed and isinstance(parsed["claims"], list):
        claims = parsed["claims"]
    
    # Format 2: Direct list (shouldn't happen with json_object but just in case)
    elif isinstance(parsed, list):
        claims = parsed
    
    # Format 3: Some other dict — find the first list value
    else:
        claims = None
        for value in parsed.values():
            if isinstance(value, list):
                claims = value
                break
        
        # Format 4: No list found — build claim from dict values manually
        if claims is None:
            text_values = [v for v in parsed.values() if isinstance(v, str)]
            if text_values:
                claims = [{"claim_id": i+1, "text": t} for i, t in enumerate(text_values)]
            else:
                # Last resort — treat entire answer as one claim
                claims = [{"claim_id": 1, "text": answer}]
    
    # Validate each claim has required fields
    validated = []
    for i, claim in enumerate(claims):
        if isinstance(claim, dict) and "text" in claim:
            validated.append({
                "claim_id": claim.get("claim_id", i+1),
                "text":     claim["text"]
            })
        elif isinstance(claim, str):
            # Some models return list of strings
            validated.append({
                "claim_id": i+1,
                "text":     claim
            })
    
    return validated if validated else [{"claim_id": 1, "text": answer}]

def split_into_sentences(docs: list[str]) -> list[dict]:
    all_sentences = []
    for doc_idx, doc in enumerate(docs):
        for sent in nltk.sent_tokenize(doc.strip()):
            sent = sent.strip()
            if len(sent) > 10:
                all_sentences.append({"sentence": sent, "doc_index": doc_idx})
    return all_sentences

def build_faiss_index(sentences: list[dict]):
    texts      = [s["sentence"] for s in sentences]
    embeddings = embed_model.encode(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index      = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, sentences

def retrieve_span(claim: str, index, sentence_store: list[dict], top_k=5) -> dict:
    claim_emb = embed_model.encode([claim])
    claim_emb = claim_emb / np.linalg.norm(claim_emb, axis=1, keepdims=True)
    scores, indices = index.search(claim_emb.astype('float32'), top_k)
    best = sentence_store[indices[0][0]]
    return {
        "sentence":         best["sentence"],
        "doc_index":        best["doc_index"],
        "similarity_score": round(float(scores[0][0]), 4),
        "low_confidence":   float(scores[0][0]) < 0.70,
        "top_3_spans": [
            {
                "sentence":  sentence_store[indices[0][i]]["sentence"],
                "doc_index": sentence_store[indices[0][i]]["doc_index"],
                "score":     round(float(scores[0][i]), 4)
            }
            for i in range(top_k)
        ]
    }

def check_entailment(claim: str, span: str) -> dict:
    result     = nli_model(f"{span} [SEP] {claim}")
    label      = result[0]["label"].upper()
    confidence = round(result[0]["score"], 4)
    label_map  = {
        "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "LABEL_0": "CONTRADICTION",
        "LABEL_1": "NEUTRAL", "LABEL_2": "ENTAILMENT"
    }
    return {"label": label_map.get(label, label), "confidence": confidence}

def rerank_with_nli(claim: str, top_spans: list[dict]) -> dict:
    all_results = []
    for rank, span_dict in enumerate(top_spans):
        nli = check_entailment(claim, span_dict["sentence"])
        all_results.append({
            "rank":           rank + 1,
            "span":           span_dict["sentence"],
            "doc_index":      span_dict["doc_index"],
            "retrieval_score": span_dict["score"],
            "nli_label":      nli["label"],
            "nli_confidence": nli["confidence"]
        })
    for label in ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]:
        matches = [r for r in all_results if r["nli_label"] == label]
        if matches:
            best = max(matches, key=lambda x: x["nli_confidence"])
            break
    return {
        "best_span":      best["span"],
        "doc_index":      best["doc_index"],
        "label":          best["nli_label"],
        "confidence":     best["nli_confidence"],
        "retrieval_rank": best["rank"],
        "reranked":       best["rank"] != 1
    }


# ── ROUTES ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "VeriFaith API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs"
    }

@app.get("/health")
def health():
    return {
        "status":    "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "claim_extractor": "groq/llama-3.3-70b-versatile",
            "span_retriever":  "sentence-transformers/all-MiniLM-L6-v2",
            "nli_checker":     "cross-encoder/nli-deberta-v3-base"
        }
    }

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    """
    Main VeriFaith evaluation endpoint.
    
    Takes a RAG-generated answer and source documents.
    Returns a faithfulness score with full claim-level breakdown.
    """
    
    try:
        # Step 1 — Extract claims
        raw_claims = extract_claims(request.answer)
        if not raw_claims:
            raise HTTPException(
                status_code=422,
                detail="Could not extract claims from the provided answer."
            )

        # Step 2 — Build index
        sentences             = split_into_sentences(request.source_docs)
        faiss_index, sent_store = build_faiss_index(sentences)

        # Step 3 — Check each claim
        claim_results        = []
        low_confidence_flags = []

        for raw_claim in raw_claims:
            retrieval  = retrieve_span(raw_claim["text"], faiss_index, sent_store)
            nli_result = rerank_with_nli(raw_claim["text"], retrieval["top_3_spans"])

            if retrieval["low_confidence"]:
                low_confidence_flags.append(raw_claim["text"])

            label_map = {
                "ENTAILMENT":    "supported",
                "NEUTRAL":       "unverified",
                "CONTRADICTION": "contradicted"
            }

            claim_results.append({
                "claim_id":      raw_claim["claim_id"],
                "claim_text":    raw_claim["text"],
                "matched_span":  nli_result["best_span"],
                "doc_index":     nli_result["doc_index"],
                "label":         nli_result["label"],
                "status":        label_map[nli_result["label"]],
                "confidence":    nli_result["confidence"],
                "retrieval_score": retrieval["similarity_score"],
                "was_reranked":  nli_result["reranked"]
            })

        # Step 4 — Compute score
        total        = len(claim_results)
        supported    = sum(1 for r in claim_results if r["label"] == "ENTAILMENT")
        neutral      = sum(1 for r in claim_results if r["label"] == "NEUTRAL")
        contradicted = sum(1 for r in claim_results if r["label"] == "CONTRADICTION")
        score        = round(supported / total, 4) if total > 0 else 0.0

        verdict = (
            "FAITHFUL"   if score >= 0.75 else
            "PARTIAL"    if score >= 0.40 else
            "UNFAITHFUL"
        )

        # Step 5 — Build response
        claim_breakdown = None
        if request.include_full_report:
            claim_breakdown = [
                ClaimResultResponse(
                    claim_id      = r["claim_id"],
                    claim_text    = r["claim_text"],
                    status        = r["status"],
                    confidence    = r["confidence"],
                    evidence_span = r["matched_span"],
                    source_doc    = r["doc_index"],
                    was_reranked  = r["was_reranked"]
                )
                for r in claim_results
            ]

        contradiction_report = [
            {
                "claim":          r["claim_text"],
                "conflicts_with": r["matched_span"],
                "confidence":     r["confidence"],
                "source_doc":     r["doc_index"]
            }
            for r in claim_results if r["label"] == "CONTRADICTION"
        ]

        return EvaluateResponse(
            verifaith_version    = "1.0.0",
            evaluated_at         = datetime.now().isoformat(),
            faithfulness_score   = score,
            verdict              = verdict,
            total_claims         = total,
            supported            = supported,
            unverified           = neutral,
            contradicted         = contradicted,
            has_contradictions   = contradicted > 0,
            claim_breakdown      = claim_breakdown,
            contradiction_report = contradiction_report,
            warnings             = {"low_confidence_retrievals": low_confidence_flags}
        )

    except HTTPException:
        raise
    except Exception as e:
        # Print full traceback to uvicorn terminal
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))