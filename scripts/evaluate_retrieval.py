import argparse
import csv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_db" / "olist_reviews"
CASES_FILE = PROJECT_ROOT / "evaluation" / "retrieval_cases.csv"
COLLECTION_NAME = "olist_reviews"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


def load_cases():
    with CASES_FILE.open(encoding="utf-8-sig", newline="") as source:
        cases = list(csv.DictReader(source))
    for case in cases:
        case["relevant_ids"] = {
            review_id.strip()
            for review_id in case["relevant_review_ids"].split(";")
            if review_id.strip()
        }
    return cases


def recall_at_k(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Avalia a recuperação Chroma com Recall@k em casos revisados manualmente."
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-distance", type=float, default=0.80)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k precisa ser maior que zero")
    if args.max_distance <= 0:
        parser.error("--max-distance precisa ser maior que zero")
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(f"Índice Chroma não encontrado: {VECTOR_DIR}")
    if not CASES_FILE.exists():
        raise FileNotFoundError(f"Casos de avaliação não encontrados: {CASES_FILE}")

    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )
    cases = load_cases()
    recalls = []
    threshold_recalls = []
    print(f"Casos: {len(cases)} | Recall@{args.top_k} | distância máxima: {args.max_distance:.2f}")

    for number, case in enumerate(cases, start=1):
        results = store.similarity_search_with_score(case["question"], k=args.top_k)
        ranked_ids = [doc.metadata.get("review_id", "") for doc, _ in results]
        accepted_ids = [
            doc.metadata.get("review_id", "")
            for doc, distance in results
            if float(distance) <= args.max_distance
        ]
        relevant_ids = case["relevant_ids"]
        recall = recall_at_k(ranked_ids, relevant_ids)
        threshold_recall = recall_at_k(accepted_ids, relevant_ids)
        recalls.append(recall)
        threshold_recalls.append(threshold_recall)
        found = set(ranked_ids) & relevant_ids
        missing = sorted(relevant_ids - found)
        print(
            f"\n{number}. Recall@{args.top_k}: {recall:.1%} | "
            f"com limite {args.max_distance:.2f}: {threshold_recall:.1%}"
        )
        print(f"   Pergunta: {case['question']}")
        print(f"   IDs relevantes encontrados: {len(found)}/{len(relevant_ids)}")
        if missing:
            print(f"   IDs de referência não recuperados no Top-{args.top_k}: {', '.join(missing)}")

    if not cases:
        raise ValueError("O conjunto de avaliação está vazio.")
    count = len(cases)
    print("\nMédia do conjunto:")
    print(f"Recall@{args.top_k}: {sum(recalls) / count:.1%}")
    print(f"Recall@{args.top_k} após limite de distância: {sum(threshold_recalls) / count:.1%}")
    print("Os casos e IDs relevantes foram revisados manualmente; este conjunto pequeno é uma validação inicial.")


if __name__ == "__main__":
    main()
