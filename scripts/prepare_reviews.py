import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "olist" / "olist_order_reviews_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "olist_reviews_prepared.csv"
TEXT_COLUMNS = ("review_comment_title", "review_comment_message")


def clean_text(value: str | None) -> str:
    return (value or "").strip()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {INPUT_FILE}")

    rows_read = 0
    empty_rows = 0
    duplicate_rows = 0
    rows_by_review_id: dict[str, dict[str, str]] = {}

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        original_columns = reader.fieldnames or []
        missing = set(TEXT_COLUMNS) - set(original_columns)
        if missing:
            raise ValueError(f"Colunas de texto ausentes: {', '.join(sorted(missing))}")
        if "review_id" not in original_columns:
            raise ValueError("Coluna obrigatória ausente: review_id")
        if "retrieval_text" in original_columns:
            raise ValueError("O arquivo de entrada já contém uma coluna retrieval_text")

        for row in reader:
            rows_read += 1
            title = clean_text(row.get("review_comment_title"))
            comment = clean_text(row.get("review_comment_message"))
            if not title and not comment:
                empty_rows += 1
                continue

            review_id = clean_text(row.get("review_id"))
            if not review_id:
                raise ValueError(f"review_id vazio na linha {rows_read + 1}")

            row["review_id"] = review_id
            row["review_comment_title"] = title
            row["review_comment_message"] = comment
            parts = []
            if title:
                parts.append(f"Título: {title}")
            if comment:
                parts.append(f"Comentário: {comment}")
            row["retrieval_text"] = "\n".join(parts)

            # Repetições observadas têm o mesmo texto, nota e datas; mantemos
            # a última associação de pedido para alinhar com o índice existente.
            if review_id in rows_by_review_id:
                duplicate_rows += 1
            rows_by_review_id[review_id] = row

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [*original_columns, "retrieval_text"]
    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows_by_review_id.values())

    rows_written = len(rows_by_review_id)
    print("~~~~~~~~~ Tratando os dados ~~~~~~~~~")
    print("Linhas mantidas somente quando título ou comentário contém texto.")
    print("Duplicatas de review_id removidas; mantida a última linha de cada ID.")
    print(f"Avaliações lidas: {rows_read:,}")
    print(f"Avaliações sem título e comentário descartadas: {empty_rows:,}")
    print(f"Linhas repetidas por review_id descartadas: {duplicate_rows:,}")
    print(f"Avaliações preparadas (review_id únicos): {rows_written:,}")
    print(f"Colunas originais preservadas: {len(original_columns)}")
    print("Coluna derivada adicionada: retrieval_text")
    print(f"Arquivo preparado: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
