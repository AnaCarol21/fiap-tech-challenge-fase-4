from pathlib import Path
import argparse
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "olist_reviews_prepared.csv"


def load_reviews():
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(
        description="Busca avaliações da Olist usando TF-IDF como baseline lexical."
    )
    parser.add_argument("question", help="Pergunta ou tema para buscar nas avaliações")
    parser.add_argument("--top-k", type=int, default=5, help="Quantidade de resultados (padrão: 5)")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k precisa ser maior que zero")
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Base preparada não encontrada em {DATA_FILE}. Rode scripts/prepare_reviews.py primeiro."
        )

    reviews = load_reviews()
    texts = [row["retrieval_text"] for row in reviews]
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
    )
    review_vectors = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([args.question])
    similarities = cosine_similarity(question_vector, review_vectors).ravel()
    best_indices = similarities.argsort()[-args.top_k:][::-1]

    print(f"Pergunta: {args.question}")
    print("Busca: TF-IDF (baseline lexical)")
    for rank, index in enumerate(best_indices, start=1):
        row = reviews[index]
        print(f"\n{rank}. Similaridade: {similarities[index]:.4f}")
        print(f"   review_id: {row['review_id']} | nota: {row['review_score']} | data: {row['review_creation_date']}")
        if row["review_comment_title"]:
            print(f"   Título: {row['review_comment_title']}")
        if row["review_comment_message"]:
            print(f"   Comentário: {row['review_comment_message']}")


if __name__ == "__main__":
    main()
