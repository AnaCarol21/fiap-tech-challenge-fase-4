import argparse
import csv
import unicodedata
from pathlib import Path

import numpy as np
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "olist_reviews_prepared.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_TOPIC_COUNTS = [5, 10, 15, 20, 30]
ASSIGNMENT_TOPIC_COUNT = 10
RANDOM_STATE = 42
MAX_ITERATIONS = 300


def load_reviews() -> list[dict[str, str]]:
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def normalize_stop_word(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def get_portuguese_stop_words() -> list[str]:
    try:
        words = set(stopwords.words("portuguese"))
    except LookupError as error:
        raise RuntimeError(
            "O corpus stopwords do NLTK não está instalado. Execute uma vez: "
            ".\\.venv\\Scripts\\python.exe -m nltk.downloader stopwords"
        ) from error

    # Mantemos negações porque são importantes para distinguir reclamações e elogios.
    words = {normalize_stop_word(word) for word in words}
    protected_words = {"nao", "nunca", "sem", "nem", "nada", "jamais"}
    words.difference_update(protected_words)
    return sorted(words)


def build_topic_text(review: dict[str, str]) -> str:
    # Usa o título e o comentário originais, sem os rótulos artificiais do retrieval_text.
    return " ".join(
        value.strip()
        for value in (
            review.get("review_comment_title", ""),
            review.get("review_comment_message", ""),
        )
        if value and value.strip()
    )


def get_output_paths(
    min_score: int | None,
    max_score: int | None,
) -> tuple[Path, Path, Path]:
    suffix = ""
    if min_score is not None or max_score is not None:
        lower = str(min_score) if min_score is not None else "any"
        upper = str(max_score) if max_score is not None else "any"
        suffix = f"_score_{lower}_{upper}"

    return (
        OUTPUT_DIR / f"nmf_topic_model_selection{suffix}.csv",
        OUTPUT_DIR / f"nmf_topic_details{suffix}.csv",
        OUTPUT_DIR / f"nmf_topic_assignments{suffix}.csv",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compara modelos NMF com TF-IDF para descobrir temas nas avaliações da Olist."
        )
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        type=int,
        default=DEFAULT_TOPIC_COUNTS,
        help="Números de tópicos a comparar (padrão: 5 10 15 20 30)",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=10,
        help="Palavras exibidas por tópico (padrão: 10)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        choices=range(1, 6),
        help="Nota mínima incluída (1 a 5)",
    )
    parser.add_argument(
        "--max-score",
        type=int,
        choices=range(1, 6),
        help="Nota máxima incluída (1 a 5)",
    )
    args = parser.parse_args()

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Base preparada não encontrada: {DATA_FILE}. "
            "Rode scripts/prepare_reviews.py primeiro."
        )
    if not args.topics or any(topic_count < 2 for topic_count in args.topics):
        parser.error("Cada quantidade de tópicos precisa ser pelo menos 2")
    if args.top_words < 2:
        parser.error("--top-words precisa ser pelo menos 2")
    if (
        args.min_score is not None
        and args.max_score is not None
        and args.min_score > args.max_score
    ):
        parser.error("--min-score não pode ser maior que --max-score")

    all_reviews = load_reviews()
    reviews = [
        review
        for review in all_reviews
        if (args.min_score is None or int(review["review_score"]) >= args.min_score)
        and (args.max_score is None or int(review["review_score"]) <= args.max_score)
    ]
    if not reviews:
        raise ValueError("Nenhuma avaliação corresponde ao filtro de nota selecionado")

    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        stop_words=get_portuguese_stop_words(),
    )
    topic_texts = [build_topic_text(review) for review in reviews]
    tfidf_matrix = vectorizer.fit_transform(topic_texts)
    active_mask = np.asarray(tfidf_matrix.getnnz(axis=1) > 0).ravel()
    active_reviews = [review for review, active in zip(reviews, active_mask) if active]
    active_texts = [text for text, active in zip(topic_texts, active_mask) if active]
    active_matrix = tfidf_matrix[active_mask]
    analyzer = vectorizer.build_analyzer()
    tokenized_texts = [analyzer(text) for text in active_texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    feature_names = vectorizer.get_feature_names_out()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection_file, topics_file, assignments_file = get_output_paths(
        args.min_score, args.max_score
    )
    selection_rows: list[dict[str, object]] = []
    topic_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []

    print(f"Avaliações na base preparada: {len(all_reviews):,}")
    print(f"Avaliações após filtro de nota: {len(reviews):,}")
    print(f"Avaliações com termos após pré-processamento: {len(active_reviews):,}")
    print(f"Vocabulário TF-IDF: {active_matrix.shape[1]:,} termos")
    if args.min_score is not None or args.max_score is not None:
        print(f"Filtro de nota: mínimo={args.min_score}, máximo={args.max_score}")
    print(f"Configurações NMF a comparar: {args.topics}")
    print("A coerência c_v é uma métrica auxiliar; os termos devem ser inspecionados.")

    for topic_count in args.topics:
        model = NMF(
            n_components=topic_count,
            init="nndsvda",
            random_state=RANDOM_STATE,
            max_iter=MAX_ITERATIONS,
        )
        document_topic_weights = model.fit_transform(active_matrix)
        dominant_topics = document_topic_weights.argmax(axis=1)
        counts = np.bincount(dominant_topics, minlength=topic_count)

        topic_word_lists = []
        topic_terms_by_id = []
        for component in model.components_:
            top_indices = component.argsort()[-args.top_words :][::-1]
            terms = [str(feature_names[index]) for index in top_indices]
            topic_terms_by_id.append(terms)
            topic_word_lists.append(terms)

        coherence = CoherenceModel(
            topics=topic_word_lists,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        ).get_coherence()

        selection_rows.append(
            {
                "n_topics": topic_count,
                "coherence_c_v": f"{coherence:.4f}",
                "reconstruction_error": f"{model.reconstruction_err_:.4f}",
                "iterations": model.n_iter_,
            }
        )

        for topic_id, terms in enumerate(topic_terms_by_id):
            member_indices = np.flatnonzero(dominant_topics == topic_id)
            review_count = int(counts[topic_id])
            representative = None
            if member_indices.size:
                best_member = member_indices[
                    document_topic_weights[member_indices, topic_id].argmax()
                ]
                representative = active_reviews[int(best_member)]

            topic_rows.append(
                {
                    "n_topics": topic_count,
                    "topic_id": topic_id,
                    "dominant_review_count": review_count,
                    "dominant_review_share_pct": (
                        f"{review_count / len(active_reviews) * 100:.2f}"
                    ),
                    "top_terms": " | ".join(terms),
                    "example_review_id": (
                        representative.get("review_id", "") if representative else ""
                    ),
                    "example_review_score": (
                        representative.get("review_score", "") if representative else ""
                    ),
                    "example_review_text": (
                        build_topic_text(representative) if representative else ""
                    ),
                }
            )

        if topic_count == ASSIGNMENT_TOPIC_COUNT:
            for review, text, topic_id, weights in zip(
                active_reviews, active_texts, dominant_topics, document_topic_weights
            ):
                assignment_rows.append(
                    {
                        "review_id": review["review_id"],
                        "review_score": review["review_score"],
                        "dominant_topic_id": int(topic_id),
                        "dominant_topic_weight": f"{weights[topic_id]:.6f}",
                        "review_text": text,
                    }
                )
            assignment_rows.sort(
                key=lambda row: (
                    int(row["dominant_topic_id"]),
                    -float(row["dominant_topic_weight"]),
                )
            )

        print(
            f"K={topic_count}: coerência c_v={coherence:.4f}; "
            f"erro de reconstrução={model.reconstruction_err_:.4f}; "
            f"iterações={model.n_iter_}",
            flush=True,
        )

    write_csv(selection_file, selection_rows)
    write_csv(topics_file, topic_rows)
    if assignment_rows:
        write_csv(assignments_file, assignment_rows)
        print(
            f"Atribuições por avaliação para K={ASSIGNMENT_TOPIC_COUNT} "
            f"salvas em: {assignments_file}"
        )
    else:
        print(
            f"Para exportar atribuições por avaliação, inclua K={ASSIGNMENT_TOPIC_COUNT} "
            "no argumento --topics."
        )

    print(f"\nComparação de K salva em: {selection_file}")
    print(f"Termos e exemplos por tópico salvos em: {topics_file}")
    print(
        "A contagem atribui cada avaliação ao seu tópico dominante; "
        "revise os termos e exemplos antes de nomear os temas."
    )


if __name__ == "__main__":
    main()
