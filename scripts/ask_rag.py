import argparse, csv, re, unicodedata
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "data" / "vector_db" / "olist_reviews"
DETAILS = ROOT / "data" / "processed" / "nmf_topic_details_score_1_2.csv"
ASSIGNMENTS = ROOT / "data" / "processed" / "nmf_topic_assignments_score_1_2.csv"
COLLECTION = "olist_reviews"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3:4b"
LABELS = {
    0: "não recebido / não chegou", 1: "não recebido / não chegou",
    2: "não recebido / não chegou", 3: "não recebido / não chegou",
    4: "pedido incompleto (faltam itens)",
    5: "produto não recebido ou produto diferente",
    6: "atraso ou demora na entrega",
    7: "insatisfação geral, qualidade ou atendimento",
    8: "produto com defeito, peça faltando ou produto divergente",
    9: "pedido de reembolso",
}


def normalized(text):
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def needs_topics(question):
    q = normalized(question)
    return any(word in q for word in (
        "frequenc", "frequent", "principais problema", "mais comum",
        "mais aparecem", "recorrente", "insatisfeit", "reclamac",
    ))


def relevant_topic_ids(question):
    q = normalized(question)
    if any(word in q for word in ("entrega", "atraso", "prazo", "chegou")):
        return {0, 1, 2, 3, 6}
    if any(word in q for word in ("qualidade", "defeito", "produto")):
        return {5, 7, 8}
    if any(word in q for word in ("reembolso", "devolucao", "estorno")):
        return {9}
    return set(range(10))


def load_topic_analysis(question):
    if not DETAILS.exists():
        return None
    topics = {}
    with DETAILS.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["n_topics"]) == 10:
                topics[int(row["topic_id"])] = row
    if len(topics) != 10:
        return None

    relevant = relevant_topic_ids(question)
    grouped = {}
    for topic_id, row in topics.items():
        if topic_id not in relevant:
            continue
        label = LABELS[topic_id]
        grouped.setdefault(label, []).append((topic_id, row))
    total = sum(int(row["dominant_review_count"]) for row in topics.values())
    delay_example = None
    if 6 in relevant and ASSIGNMENTS.exists():
        with ASSIGNMENTS.open(encoding="utf-8-sig", newline="") as f:
            candidates = [
                row for row in csv.DictReader(f)
                if row["dominant_topic_id"] == "6"
                and re.search(r"atras|demor|prazo|vagaros|lento", row["review_text"], re.I)
            ]
        if candidates:
            delay_example = max(
                candidates, key=lambda row: float(row["dominant_topic_weight"])
            )
    lines = [
        "Análise NMF de avaliações com nota 1 ou 2 e termos após o pré-processamento. Cada avaliação desse grupo foi atribuída "
        "a um tópico dominante; o grupo 'não recebido / não chegou' soma os "
        "tópicos 0–3. Os rótulos são interpretações provisórias, e as contagens "
        "não equivalem a marcação manual exata de menções. Percentuais usam como "
        f"denominador as {total:,} avaliações com termos após o pré-processamento."
    ]
    for label, entries in grouped.items():
        count = sum(int(row["dominant_review_count"]) for _, row in entries)
        lines.append(f"\nTema: {label} — {count:,} avaliações ({count / total:.1%}).")
        for topic_id, row in entries:
            example_id = row["example_review_id"]
            example_text = row["example_review_text"]
            if topic_id == 6 and delay_example:
                example_id = delay_example["review_id"]
                example_text = delay_example["review_text"]
            lines.append(
                f"Exemplo representativo do tópico {topic_id}, "
                f"review_id {example_id}: {example_text}"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Responde usando avaliações da Olist.")
    parser.add_argument("question", help="Pergunta sobre as avaliações")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    use_topics = needs_topics(args.question)
    analysis = load_topic_analysis(args.question) if use_topics else None
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(VECTOR_DIR),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )
    search = {"k": args.top_k}
    if use_topics:
        search["filter"] = {"review_score": {"$lte": 2}}
    docs = store.similarity_search(args.question, **search)
    if not docs:
        print("Nenhuma avaliação relevante foi encontrada no índice.")
        return

    sources = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        sources.append(
            f"Fonte recuperada {i} | review_id: {meta.get('review_id')} | nota: "
            f"{meta.get('review_score')} | data: {meta.get('review_creation_date')}\n"
            f"{doc.page_content}"
        )
    context = "\n\n---\n\n".join(sources)
    if analysis:
        aggregate = "\n\nAnálise NMF e exemplos representativos:\n" + analysis
        instruction = (
            "Apresente todas as categorias da análise NMF relacionadas à pergunta, "
            "sem omitir nenhuma, em ordem decrescente de contagem. Para cada uma, "
            "informe contagem, percentual e cite o review_id do exemplo representativo "
            "daquela categoria. Em seguida, resuma o tema de maior frequência. "
            "Esclareça que são avaliações com nota 1 ou 2, cada avaliação foi contada "
            "em um tópico dominante, os rótulos são provisórios e as contagens não "
            "são menções exatas nem representam toda a base. As fontes recuperadas "
            "são exemplos adicionais."
        )
    else:
        aggregate = ""
        instruction = (
            "As fontes recuperadas são exemplos próximos da pergunta, não a base "
            "inteira nem amostra estatística. Sem contagens globais, não afirme "
            "frequência ou prevalência; descreva apenas os exemplos."
        )

    prompt = f"""Você analisa avaliações de clientes de comércio eletrônico.
Responda em português e use somente informações do contexto.
{instruction}
Não invente causas ou detalhes. Se faltar evidência, diga isso.
Ao citar uma avaliação, copie seu review_id exatamente.

Pergunta: {args.question}
{aggregate}

Fontes recuperadas por similaridade ({len(docs)} avaliações):
{context}

Resposta:"""
    response = ChatOllama(model=CHAT_MODEL, temperature=0).invoke(prompt)
    print(f"Pergunta: {args.question}\n")
    if analysis:
        print("Resultado quantitativo da análise NMF (estimativas):")
        print(analysis + "\n")
        print(f"Explicação gerada pelo modelo ({CHAT_MODEL}):\n{response.content}\n")
    else:
        print(f"Resposta ({CHAT_MODEL}):\n{response.content}\n")
    print("Fontes recuperadas por similaridade:")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"\nFonte {i} | review_id: {meta.get('review_id')} | nota: "
              f"{meta.get('review_score')} | data: {meta.get('review_creation_date')}")
        print(f"   {doc.page_content}")


if __name__ == "__main__":
    main()
