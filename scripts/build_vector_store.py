import csv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "olist_reviews_prepared.csv"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_db" / "olist_reviews"
COLLECTION_NAME = "olist_reviews"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
BATCH_SIZE = 64


def load_documents() -> list[Document]:
    documents = []
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            metadata = {
                "review_id": row["review_id"],
                "order_id": row["order_id"],
                "review_score": int(row["review_score"]),
                "review_creation_date": row["review_creation_date"],
                "review_answer_timestamp": row["review_answer_timestamp"],
            }
            documents.append(
                Document(page_content=row["retrieval_text"], metadata=metadata)
            )
    return documents


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Base preparada não encontrada: {DATA_FILE}. "
            "Rode scripts/prepare_reviews.py primeiro."
        )

    documents = load_documents()
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )

    existing_ids = set(vector_store.get().get("ids", []))
    new_documents = [
        document for document in documents
        if document.metadata["review_id"] not in existing_ids
    ]
    total_batches = (len(new_documents) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_number, start in enumerate(range(0, len(new_documents), BATCH_SIZE), start=1):
        batch = new_documents[start : start + BATCH_SIZE]
        vector_store.add_documents(
            documents=batch,
            ids=[document.metadata["review_id"] for document in batch],
        )
        print(
            f"Lote {batch_number}/{total_batches}: {len(batch)} avaliações indexadas",
            flush=True,
        )

    indexed_count = vector_store._collection.count()
    print(f"Avaliações preparadas: {len(documents):,}")
    print(f"Novas avaliações indexadas nesta execução: {len(new_documents):,}")
    print(f"Total de avaliações no Chroma: {indexed_count:,}")
    print(f"Modelo local de embeddings: {EMBEDDING_MODEL}")
    print(f"Índice persistido em: {VECTOR_DIR}")


if __name__ == "__main__":
    main()
