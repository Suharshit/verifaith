from verifaith.claim_extractor import extract_claims
from verifaith.span_retriever import split_into_sentences, build_faiss_index, retrieve_span
from verifaith.entailment_checker import rerank_with_nli
from verifaith.models import ClaimResult, VeriFaithResult

def evaluate(answer: str, source_docs: list[str]) -> VeriFaithResult:
    
    raw_claims = extract_claims(answer)
    sentences = split_into_sentences(source_docs)
    faiss_index, sentence_store = build_faiss_index(sentences)
    
    claim_results = []
    low_confidence_flags = []
    
    for raw_claim in raw_claims:
        retrieval  = retrieve_span(raw_claim["text"], faiss_index, sentence_store)
        nli_result = rerank_with_nli(raw_claim["text"], retrieval["top_3_spans"])
        
        if retrieval["low_confidence"]:
            low_confidence_flags.append(raw_claim["text"])
        
        claim_results.append(ClaimResult(
            claim_id        = raw_claim["claim_id"],
            claim_text      = raw_claim["text"],
            matched_span    = nli_result["best_span"],
            doc_index       = nli_result["doc_index"],
            label           = nli_result["label"],
            confidence      = nli_result["confidence"],
            retrieval_score = retrieval["similarity_score"],
            was_reranked    = nli_result["reranked"]
        ))
    
    total        = len(claim_results)
    supported    = sum(1 for r in claim_results if r.label == "ENTAILMENT")
    neutral      = sum(1 for r in claim_results if r.label == "NEUTRAL")
    contradicted = sum(1 for r in claim_results if r.label == "CONTRADICTION")
    
    return VeriFaithResult(
        faithfulness_score   = round(supported / total, 4) if total > 0 else 0.0,
        total_claims         = total,
        supported_count      = supported,
        neutral_count        = neutral,
        contradiction_count  = contradicted,
        per_claim_report     = claim_results,
        contradiction_report = [r for r in claim_results if r.label == "CONTRADICTION"],
        low_confidence_flags = low_confidence_flags
    )