from collections import Counter
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEWS_FILE = PROJECT_ROOT / "data" / "raw" / "olist" / "olist_order_reviews_dataset.csv"
REQUIRED_COLUMNS = {
    "review_id",
    "order_id",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
}


def has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def main() -> None:
    if not REVIEWS_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {REVIEWS_FILE}")

    with REVIEWS_FILE.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Colunas esperadas ausentes: {', '.join(sorted(missing))}")

        total = 0
        title_count = 0
        comment_count = 0
        eligible_count = 0
        scores_with_text: Counter[str] = Counter()

        for row in reader:
            total += 1
            title_filled = has_text(row.get("review_comment_title"))
            comment_filled = has_text(row.get("review_comment_message"))
            title_count += title_filled
            comment_count += comment_filled
            if title_filled or comment_filled:
                eligible_count += 1
                scores_with_text[row["review_score"]] += 1
    print("~~~~~~~~~ Analisando quantidade de dados presentes ~~~~~~~~~")
    print(f"Arquivo: {REVIEWS_FILE}")
    print(f"Linhas totais: {total:,}")
    print(f"Títulos preenchidos: {title_count:,}")
    print(f"Comentários preenchidos: {comment_count:,}")
    print(f"Linhas mantidas (título OU comentário): {eligible_count:,}")
    print(f"Linhas descartadas (ambos vazios): {total - eligible_count:,}")
    print("Notas entre as linhas mantidas:")
    for score, count in sorted(scores_with_text.items()):
        print(f"  {score}: {count:,}")


if __name__ == "__main__":
    main()
