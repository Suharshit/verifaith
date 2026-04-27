from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-base",
    device=device
)

def check_entailment(claim: str, span: str) -> dict:
    nli_input = f"{span} [SEP] {claim}"
    result = nli_model(nli_input)
    label = result[0]["label"].upper()
    confidence = round(result[0]["score"], 4)
    label_map = {
        "ENTAILMENT": "ENTAILMENT",
        "NEUTRAL": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION",
        "LABEL_0": "CONTRADICTION",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "ENTAILMENT"
    }
    return {
        "label": label_map.get(label, label),
        "confidence": confidence,
        "claim": claim,
        "span": span
    }

def rerank_with_nli(claim: str, top_spans: list[dict]) -> dict:
    all_results = []
    for rank, span_dict in enumerate(top_spans):
        nli_result = check_entailment(claim, span_dict["sentence"])
        all_results.append({
            "rank": rank + 1,
            "span": span_dict["sentence"],
            "doc_index": span_dict["doc_index"],
            "retrieval_score": span_dict["score"],
            "nli_label": nli_result["label"],
            "nli_confidence": nli_result["confidence"]
        })

    # Priority: CONTRADICTION → ENTAILMENT → NEUTRAL
    for label in ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]:
        matches = [r for r in all_results if r["nli_label"] == label]
        if matches:
            best = max(matches, key=lambda x: x["nli_confidence"])
            break

    return {
        "best_span": best["span"],
        "doc_index": best["doc_index"],
        "label": best["nli_label"],
        "confidence": best["nli_confidence"],
        "retrieval_rank": best["rank"],
        "reranked": best["rank"] != 1,
        "all_results": all_results
    }