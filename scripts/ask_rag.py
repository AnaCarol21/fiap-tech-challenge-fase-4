import argparse, csv, re, unicodedata
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "data" / "vector_db" / "olist_reviews"
DETAILS = ROOT / "data" / "processed" / "nmf_topic_details_score_1_2.csv"
ASSIGNMENTS = ROOT / "data" / "processed" / "nmf_topic_assignments_score_1_2.csv"
POSITIVE_DETAILS = ROOT / "data" / "processed" / "nmf_topic_details_score_4_5.csv"
POSITIVE_ASSIGNMENTS = ROOT / "data" / "processed" / "nmf_topic_assignments_score_4_5.csv"
COLLECTION = "olist_reviews"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3:4b"
DEFAULT_MAX_DISTANCE = 0.80
LABELS = {
    0: "não recebido / não chegou", 1: "não recebido / não chegou",
    2: "não recebido / não chegou", 3: "não recebido / não chegou",
    4: "pedido incompleto (faltam itens)",
    5: "produto não recebido ou produto diferente",
    6: "atraso ou demora na entrega",
    7: "insatisfação e recomendação em avaliações mistas",
    8: "produto com defeito, peça faltando ou produto divergente",
    9: "pedido de reembolso",
}


def normalized(text):
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def remove_review_ids(text):
    text = re.sub(
        r"\s*\(?\s*review_id\s*:\s*[0-9a-f]{32}\s*\)?", "", text, flags=re.I
    )
    text = re.sub(
        r"\s*\(?\s*review_id\s+[0-9a-f]{32}\s*\)?", "", text, flags=re.I
    )
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def needs_topics(question):
    q = normalized(question)
    return any(word in q for word in (
        "frequenc", "frequent", "principais problema", "mais comum",
        "mais aparecem", "recorrente", "insatisfeit", "reclamac",
    ))


def asks_positive(question):
    q = normalized(question)
    return any(term in q for term in (
        "elogi", "valoriza", "apreci", "aspectos positivos", "experiencia positiva",
    ))


def asks_exact_cause(question):
    q = normalized(question)
    return any(phrase in q for phrase in (
        "causa exata", "causa real", "motivo exato", "motivo real",
        "qual foi a causa", "qual a causa", "qual foi o motivo", "qual o motivo",
        "qual transportadora", "que transportadora", "quem causou o atraso",
    ))


def relevant_positive_topic_ids(question):
    q = normalized(question)
    if any(term in q for term in ("entrega", "prazo", "rapida", "rapido", "pontual")):
        return {2, 5}
    if any(term in q for term in ("qualidade", "defeito")):
        return {8}
    if any(term in q for term in ("recomendo", "recomend", "indicaria")):
        return {3, 9}
    return set(range(10))


def load_positive_topic_analysis(question):
    if not POSITIVE_DETAILS.exists():
        return None
    topics = {}
    with POSITIVE_DETAILS.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if int(row["n_topics"]) == 10:
                topics[int(row["topic_id"])] = row
    if len(topics) != 10:
        return None

    positive_labels = {
        0: "elogios gerais: bom",
        1: "elogios gerais: ótimo",
        2: "entrega antes do prazo",
        3: "recomendação do produto ou da loja",
        4: "elogio geral: excelente",
        5: "entrega rápida",
        6: "compra sem problemas / tudo certo",
        7: "avaliações positivas variadas sobre o produto",
        8: "qualidade do produto",
        9: "recomendação e rapidez (tema misto)",
    }
    relevant = relevant_positive_topic_ids(question)
    grouped = {}
    for topic_id, row in topics.items():
        if topic_id in relevant:
            grouped.setdefault(positive_labels[topic_id], []).append((topic_id, row))

    total = sum(int(row["dominant_review_count"]) for row in topics.values())
    quality_example = None
    if 8 in relevant and POSITIVE_ASSIGNMENTS.exists():
        with POSITIVE_ASSIGNMENTS.open(encoding="utf-8-sig", newline="") as source:
            candidates = [
                row for row in csv.DictReader(source)
                if row["dominant_topic_id"] == "8"
                and re.search(r"qualidade", row["review_text"], re.I)
            ]
        if candidates:
            quality_example = max(
                candidates, key=lambda row: float(row["dominant_topic_weight"])
            )

    lines = [
        "Análise NMF de avaliações com nota 4 ou 5 e termos após o pré-processamento. "
        "Cada avaliação desse grupo foi atribuída a um tópico dominante. Os rótulos "
        "são interpretações provisórias e as contagens não equivalem a uma marcação "
        "manual exata de menções. Percentuais usam como denominador as "
        f"{total:,} avaliações com termos após o pré-processamento."
    ]
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: sum(
            int(row["dominant_review_count"]) for _, row in item[1]
        ),
        reverse=True,
    )
    for label, entries in sorted_groups:
        count = sum(int(row["dominant_review_count"]) for _, row in entries)
        lines.append(f"\nTema: {label} — {count:,} avaliações ({count / total:.1%}).")
        for topic_id, row in entries:
            example_id = row["example_review_id"]
            example_text = row["example_review_text"]
            if topic_id == 8 and quality_example:
                example_id = quality_example["review_id"]
                example_text = quality_example["review_text"]
            lines.append(
                f"Exemplo representativo do tópico {topic_id}, "
                f"review_id {example_id}: {example_text}"
            )
    return "\n".join(lines)

def relevant_topic_ids(question):
    q = normalized(question)
    if any(word in q for word in ("entrega", "atraso", "prazo", "chegou")):
        return {0, 1, 2, 3, 6}
    if any(word in q for word in (
        "qualidade", "defeito", "peca faltando", "quebrado", "durabilidade",
    )):
        return {8}
    if any(word in q for word in ("produto", "errado", "diferente")):
        return {5, 8}
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


def remove_invalid_evidence_refs(text, source_count, nmf_count):
    def keep_valid(match):
        kind, number_text = match.groups()
        number = int(number_text)
        limit = source_count if kind.lower() == "fonte" else nmf_count
        return match.group(0) if 1 <= number <= limit else ""

    return re.sub(r"\[(Fonte|NMF)\s+(\d+)\]", keep_valid, text, flags=re.I)


def retain_cited_sentences(text):
    # Repair missing spaces between model-generated sentences before filtering.
    text = re.sub(r"([.!?])(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])", r"\1 ", text)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    cited = [
        sentence.strip()
        for sentence in sentences
        if re.search(r"\[(?:Fonte|NMF)\s+\d+\]", sentence, re.I)
    ]
    return " ".join(cited)


def soften_evidence_language(text):
    endings = {"o": "o", "a": "a", "os": "os", "as": "as"}

    def replace_proven(match):
        ending = match.group(1).lower()
        replacement = "ilustrad" + endings[ending] + " por"
        if match.group(0)[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return re.sub(
        r"\bcomprovad([oa]s?)\s+por\b",
        replace_proven,
        text,
        flags=re.I,
    )



def build_topic_context(analysis, positive_mode, max_topics=4):
    """Condensa temas NMF e os numera para permitir citações verificáveis."""
    if not analysis:
        return ""

    topic_pattern = re.compile(
        r"\nTema: (.*?) — ([\d,]+) avaliações \(([\d.]+)%\)\.\n"
        r"(.*?)(?=\nTema: |\Z)",
        re.S,
    )
    topics = []
    for match in topic_pattern.finditer(analysis):
        label, count_text, percent_text, details = match.groups()
        if not positive_mode and "insatisfação e recomendação em avaliações mistas" in label.lower():
            continue

        example_match = re.search(
            r"Exemplo representativo do tópico \d+, review_id ([0-9a-f]+): (.*?)(?=\nExemplo representativo|\Z)",
            details,
            re.S | re.I,
        )
        example_id = example_match.group(1) if example_match else ""
        example = " ".join(example_match.group(2).split()) if example_match else ""
        topics.append({
            "label": label,
            "count": int(count_text.replace(",", "")),
            "percent": percent_text,
            "example_id": example_id,
            "example": example,
        })

    topics.sort(key=lambda topic: topic["count"], reverse=True)
    score_range = "4 ou 5" if positive_mode else "1 ou 2"
    lines = [
        f"Temas NMF estimados em avaliações com nota {score_range}, em ordem "
        "decrescente de contagem. Cada avaliação foi atribuída a um tópico dominante; "
        "os rótulos são provisórios e os comentários são exemplos representativos. "
        "Use [NMF n] para citar o exemplo numerado correspondente."
    ]
    for number, topic in enumerate(topics[:max_topics], start=1):
        lines.append(f"Tema NMF {number}: {topic['label']}.")
        if topic["example"]:
            lines.append(
                f"Exemplo NMF {number} | review_id {topic['example_id']}: {topic['example']}"
            )
    return "\n".join(lines)

def build_topic_search_queries(analysis, positive_mode, max_topics=4):
    """Transforma a pergunta em buscas curtas baseadas nos temas NMF selecionados."""
    topic_context = build_topic_context(analysis, positive_mode, max_topics)
    queries = []
    current_topic = None
    for line in topic_context.splitlines():
        if line.startswith("Tema NMF "):
            current_topic = line.split(": ", 1)[1].rstrip(".")
        elif line.startswith("Exemplo NMF ") and current_topic:
            example = line.split(": ", 1)[1]
            queries.append(f"{current_topic}. {example}")
            current_topic = None
    return list(dict.fromkeys(queries))

def retrieve_documents(store, question, analysis, positive_mode, use_topics, top_k, max_distance):
    """Aplica a recuperação do RAG: filtros, expansão NMF, corte e round-robin."""
    search = {"k": top_k}
    if positive_mode:
        search["filter"] = {"review_score": {"$gte": 4}}
    elif use_topics:
        search["filter"] = {"review_score": {"$lte": 2}}
    search_queries = [question]
    if analysis:
        search_queries.extend(build_topic_search_queries(analysis, positive_mode))
    search_queries = list(dict.fromkeys(search_queries))
    candidates_by_query = []
    for search_query in search_queries:
        candidates = store.similarity_search_with_score(search_query, **search)
        best_for_query = {}
        for doc, distance in candidates:
            distance = float(distance)
            if distance > max_distance:
                continue
            review_id = doc.metadata.get("review_id")
            if review_id not in best_for_query or distance < best_for_query[review_id][1]:
                best_for_query[review_id] = (doc, distance)
        candidates_by_query.append(sorted(best_for_query.values(), key=lambda item: item[1]))
    scored_docs = []
    selected_ids = set()
    max_candidates = max((len(items) for items in candidates_by_query), default=0)
    for rank in range(max_candidates):
        for query_candidates in candidates_by_query:
            if rank >= len(query_candidates):
                continue
            doc, distance = query_candidates[rank]
            review_id = doc.metadata.get("review_id")
            if review_id in selected_ids:
                continue
            selected_ids.add(review_id)
            scored_docs.append((doc, distance))
            if len(scored_docs) >= top_k:
                break
        if len(scored_docs) >= top_k:
            break
    return search_queries, scored_docs


def main():
    parser = argparse.ArgumentParser(description="Responde usando avaliações da Olist.")
    parser.add_argument("question", help="Pergunta sobre as avaliações")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help="Distância máxima Chroma para aceitar uma fonte (padrão: 0.80)",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k precisa ser maior que zero")
    if args.max_distance <= 0:
        parser.error("--max-distance precisa ser maior que zero")

    question_text = normalized(args.question)
    asks_for_unavailable_identity = any(term in question_text for term in (
        "nome do cliente", "endereco do cliente", "telefone do cliente",
        "cpf", "quem escreveu", "quem fez a avaliacao",
    ))
    positive_mode = asks_positive(args.question) and not asks_for_unavailable_identity
    use_topics = (
        needs_topics(args.question)
        and not asks_for_unavailable_identity
        and not positive_mode
    )
    if positive_mode:
        analysis = load_positive_topic_analysis(args.question)
    else:
        analysis = load_topic_analysis(args.question) if use_topics else None
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(VECTOR_DIR),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )
    search_queries, scored_docs = retrieve_documents(
        store, args.question, analysis, positive_mode, use_topics, args.top_k, args.max_distance
    )
    docs = [doc for doc, _ in scored_docs]
    if not docs and not analysis:
        print(
            f"Nenhuma fonte ficou dentro do limite de distância "
            f"{args.max_distance:.2f}. Não há evidência suficiente para responder."
        )
        return

    sources = []
    for i, (doc, distance) in enumerate(scored_docs, 1):
        meta = doc.metadata
        sources.append(
            f"Fonte {i} | distância: {distance:.4f} "
            f"| review_id: {meta.get('review_id')} | nota: "
            f"{meta.get('review_score')} | data: {meta.get('review_creation_date')}\n"
            f"{doc.page_content}"
        )
    context = "\n\n---\n\n".join(sources)
    if not context:
        context = (
            "A busca semântica não encontrou comentários dentro do limite de "
            f"distância {args.max_distance:.2f}. Se houver análise NMF acima, "
            "ela contém apenas seus próprios exemplos representativos."
        )
    if analysis:
        model_analysis = build_topic_context(analysis, positive_mode)
        aggregate = "\n\nContexto temático selecionado para a resposta:\n" + model_analysis
        instruction = (
            "Responda diretamente à pergunta em 2 a 4 frases naturais. Para perguntas "
            "gerais sobre principais problemas, padrões ou elogios, use os temas NMF "
            "como síntese do conjunto: eles estão em ordem decrescente de contagem "
            "estimada. Os comentários recuperados pelo Chroma são exemplos adicionais "
            "e não uma amostra de prevalência; não conclua que um tema não existe só "
            "porque ele não apareceu entre as fontes recuperadas. Sintetize os temas "
            "concretos mais relevantes sem repetir a lista inteira. Não reproduza "
            "contagens, percentuais nem review_ids: o programa apresenta esses dados "
            "em uma seção separada, com valores e identificadores exatos. Cite cada "
            "afirmação factual com [Fonte n] quando apoiada por comentário Chroma ou "
            "[NMF n] quando apoiada por exemplo NMF; use somente números de referência "
            "presentes no contexto. Em avaliações negativas, não trate o tópico amplo "
            "'insatisfação e recomendação em avaliações mistas' como um problema "
            "específico. Não amplie a descrição de um comentário além do problema "
            "que ele relata claramente, mesmo quando o rótulo NMF agrupa problemas "
            "alternativos. Descreva os exemplos como ilustrações; não diga que poucos "
            "comentários provam ou confirmam a frequência geral de um tema. Prefira "
            "termos como 'indica' ou 'ilustra'. Deixe claro que os temas são estimativas "
            "NMF, com rótulos provisórios, e não contagens exatas de menções."
        )
    else:
        aggregate = ""
        instruction = (
            "As fontes recuperadas são exemplos próximos da pergunta, não a base "
            "inteira nem amostra estatística. Sem contagens globais, não afirme "
            "frequência ou prevalência; descreva apenas os exemplos. Cite cada "
            "afirmação factual com [Fonte n], usando somente referências válidas do "
            "contexto."
        )

    prompt = f"""Você analisa avaliações de clientes de comércio eletrônico.
Responda em português e use somente informações do contexto.
{instruction}
Não invente causas ou detalhes. Avaliações são relatos e opiniões de clientes, não confirmação independente dos fatos. Se a pergunta pedir causa, motivo ou responsável, atribua cada explicação ao comentário que a relata; não apresente uma percepção individual como causa confirmada nem generalize uma causa única. Preserve marcas de incerteza do comentário, como “acredito”, “penso” ou “não sei”. Um comentário que relata atraso ou defeito, sem explicar sua causa, não permite inferir o motivo. Se as fontes não trouxerem uma explicação explícita, diga que as evidências recuperadas são insuficientes; não transforme ausência de evidência em afirmação absoluta sobre toda a base.
Não escreva review_ids na resposta: use apenas as referências [Fonte n] e [NMF n].

Pergunta: {args.question}
{aggregate}

Fontes recuperadas por similaridade ({len(docs)} avaliações):
{context}

Resposta:"""
    if asks_exact_cause(args.question):
        source_refs = " ".join(f"[Fonte {number}]" for number in range(1, len(docs) + 1))
        answer = (
            "Não é possível confirmar uma causa exata a partir destas avaliações. "
            "Os comentários são relatos e percepções individuais, não uma verificação "
            f"independente da causa. Consulte os relatos recuperados abaixo. {source_refs}"
        )
    else:
        response = ChatOllama(model=CHAT_MODEL, temperature=0).invoke(prompt)
        answer = remove_review_ids(response.content)
        nmf_reference_count = len(re.findall(r"^Exemplo NMF \d+ \|", model_analysis, re.M)) if analysis else 0
        answer = remove_invalid_evidence_refs(answer, len(docs), nmf_reference_count)
        answer = soften_evidence_language(answer)
        answer = retain_cited_sentences(answer)
        if not answer:
            answer = "A resposta gerada não trouxe referências válidas. Consulte as evidências detalhadas abaixo."
    print(f"Pergunta: {args.question}\n")
    if len(search_queries) > 1:
        print("Consultas usadas na recuperação:")
        for search_query in search_queries:
            print(f"- {search_query}")
        print()
    print(f"Resposta ({CHAT_MODEL}):\n{answer}\n")
    if analysis:
        print("Detalhamento quantitativo da análise NMF (estimativas e exemplos):")
        print(analysis + "\n")
        print("Mapeamento das referências [NMF n]:")
        for line in model_analysis.splitlines():
            if line.startswith("Exemplo NMF "):
                print(line)
        print()
    print(f"Fontes recuperadas por similaridade (distância <= {args.max_distance:.2f}):")
    if not scored_docs:
        print("   Nenhuma; foram usados os exemplos representativos da análise NMF.")
    for i, (doc, distance) in enumerate(scored_docs, 1):
        meta = doc.metadata
        print(f"\nFonte {i} | distância: {distance:.4f} "
              f"| review_id: {meta.get('review_id')} | nota: {meta.get('review_score')} "
              f"| data: {meta.get('review_creation_date')}")
        print(f"   {doc.page_content}")


if __name__ == "__main__":
    main()
