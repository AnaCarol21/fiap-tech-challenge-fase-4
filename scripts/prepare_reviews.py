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

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows_read = 0
    rows_written = 0

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        original_columns = reader.fieldnames or []
        missing = set(TEXT_COLUMNS) - set(original_columns)
        if missing:
            raise ValueError(f"Colunas de texto ausentes: {', '.join(sorted(missing))}")
        if "retrieval_text" in original_columns:
            raise ValueError("O arquivo de entrada já contém uma coluna retrieval_text")

        output_columns = [*original_columns, "retrieval_text"]
        with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=output_columns)
            writer.writeheader()

            for row in reader:
                rows_read += 1
                title = clean_text(row.get("review_comment_title"))
                comment = clean_text(row.get("review_comment_message"))
                if not title and not comment:
                    continue

                row["review_comment_title"] = title
                row["review_comment_message"] = comment
                parts = []
                if title:
                    parts.append(f"Título: {title}")
                if comment:
                    parts.append(f"Comentário: {comment}")
                row["retrieval_text"] = "\n".join(parts)
                writer.writerow(row)
                rows_written += 1
    print("~~~~~~~~~ Tratando os dados ~~~~~~~~~")
    
    # Aqui optamos para remover as linhas em que tanto o título quanto o comentário estariam em branco
    # Retornando somente o que tiver conteúdo de título OU comentário
    print(f"Avaliações lidas: {rows_read:,}")
    print(f"Avaliações mantidas: {rows_written:,}")
    print(f"Avaliações sem título e comentário descartadas: {rows_read - rows_written:,}")
    print(f"Colunas originais preservadas: {len(original_columns)}")
    print("Coluna derivada adicionada: retrieval_text")
    print(f"Arquivo preparado: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
