import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_db" / "olist_reviews"
COLLECTION_NAME = "olist_reviews"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3:4b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Responde perguntas usando avaliações recuperadas da Olist."
    )
    parser.add_argument("question", help="Pergunta sobre as avaliações")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )
    retrieved_documents = vector_store.similarity_search(
        args.question,
        k=args.top_k,
    )

    if not retrieved_documents:
        print("Nenhuma avaliação relevante foi encontrada no índice.")
        return

    context_parts = []
    for position, document in enumerate(retrieved_documents, start=1):
        metadata = document.metadata
        context_parts.append(
            f"Fonte {position} | review_id: {metadata.get('review_id')} "
            f"| nota: {metadata.get('review_score')} "
            f"| data: {metadata.get('review_creation_date')}\n"
            f"{document.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Você é um assistente que analisa avaliações de clientes de comércio eletrônico.
Responda em português, usando somente informações explicitamente presentes nas fontes.
As fontes são apenas as avaliações mais próximas da pergunta recuperadas pela busca; elas não são a base inteira nem uma amostra estatística.
Não afirme que um tema é o mais frequente, recorrente, majoritário ou principal na base completa sem uma contagem global explícita no contexto.
Se a pergunta pedir frequência ou prevalência e o contexto não trouxer contagens globais, explique que a busca recupera apenas exemplos e não permite determinar isso. Você pode descrever os temas encontrados nesses exemplos e contar ocorrências somente dentro das fontes recuperadas, identificando claramente esse limite.
Não invente causas, fatos ou detalhes. Se não houver evidência suficiente, diga isso claramente.
Ao mencionar uma informação específica, cite o review_id correspondente exatamente como aparece na fonte.
Não associe uma informação a uma fonte se o texto dela não a confirmar.

Pergunta: {args.question}

Fontes disponíveis ({len(retrieved_documents)} avaliações recuperadas):
{context}

Resposta:"""

    model = ChatOllama(model=CHAT_MODEL, temperature=0)
    response = model.invoke(prompt)

    print(f"Pergunta: {args.question}\n")
    print(f"Resposta ({CHAT_MODEL}):\n{response.content}\n")
    print("Fontes recuperadas e enviadas ao modelo:")
    for position, document in enumerate(retrieved_documents, start=1):
        metadata = document.metadata
        print(
            f"\nFonte {position} | review_id: {metadata.get('review_id')} "
            f"| nota: {metadata.get('review_score')} "
            f"| data: {metadata.get('review_creation_date')}"
        )
        print(f"   {document.page_content}")


if __name__ == "__main__":
    main()
