import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_db" / "olist_reviews"
COLLECTION_NAME = "olist_reviews"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Busca avaliações semelhantes no índice Chroma da Olist."
    )
    parser.add_argument("question", help="Pergunta ou assunto para buscar")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )

    results = vector_store.similarity_search_with_score(
        args.question,
        k=args.top_k,
    )

    print(f"Pergunta: {args.question}")
    print(f"Modelo de embeddings: {EMBEDDING_MODEL}")

    for position, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        print(f"\n{position}. Distância: {distance:.4f}")
        print(
            f"   review_id: {metadata.get('review_id')} "
            f"| nota: {metadata.get('review_score')} "
            f"| data: {metadata.get('review_creation_date')}"
        )
        print(f"   Avaliação: {document.page_content}")


if __name__ == "__main__":
    main()
