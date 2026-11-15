import argparse
import csv
from pathlib import Path
from ask_rag import COLLECTION, EMBEDDING_MODEL, VECTOR_DIR, OllamaEmbeddings, Chroma, asks_positive, needs_topics, load_topic_analysis, load_positive_topic_analysis, retrieve_documents
ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"evaluation"/"retrieval_cases.csv"
JUDGMENTS=ROOT/"evaluation"/"retrieval_judgments.csv"
def load_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def main():
    parser=argparse.ArgumentParser(description="Inspeciona candidatos e calcula Precisão@k com julgamentos manuais.")
    parser.add_argument("--top-k",type=int,default=5); parser.add_argument("--max-distance",type=float,default=0.80); args=parser.parse_args()
    if args.top_k<1: parser.error("--top-k precisa ser maior que zero")
    if not VECTOR_DIR.exists(): raise FileNotFoundError(f"Índice Chroma não encontrado: {VECTOR_DIR}")
    cases=load_csv(CASES); judgments=load_csv(JUDGMENTS) if JUDGMENTS.exists() else []
    labels={(r["question"],r["review_id"]):r["relevant"].strip().lower() in {"1","sim","s","yes","y"} for r in judgments if r.get("relevant","").strip()}
    store=Chroma(collection_name=COLLECTION,persist_directory=str(VECTOR_DIR),embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)); scores=[]
    print(f"Casos: {len(cases)} | Precisão@{args.top_k} | distância máxima: {args.max_distance:.2f}")
    for idx,case in enumerate(cases,1):
        q=case["question"]; positive=asks_positive(q); topics=needs_topics(q) and not positive
        analysis=load_positive_topic_analysis(q) if positive else (load_topic_analysis(q) if topics else None)
        queries,results=retrieve_documents(store,q,analysis,positive,topics,args.top_k,args.max_distance)
        print(f"\n{idx}. {q}")
        for query in queries: print(f"   Consulta: {query}")
        complete=True; relevant=0
        for rank,(doc,distance) in enumerate(results,1):
            rid=doc.metadata.get("review_id",""); key=(q,rid)
            if key not in labels: complete=False; status="PENDENTE"
            else: is_rel=labels[key]; relevant+=int(is_rel); status="RELEVANTE" if is_rel else "NÃO RELEVANTE"
            print(f"   {rank}. [{status}] distância={distance:.4f} | id={rid} | nota={doc.metadata.get('review_score')}")
            print(f"      {doc.page_content.replace(chr(10),' ')[:500]}")
        if complete:
            score=relevant/args.top_k; scores.append(score); print(f"   Precisão@{args.top_k}: {score:.1%} ({relevant}/{args.top_k})")
        else: print("   Precisão pendente: classifique os candidatos no retrieval_judgments.csv e rode novamente.")
    if scores: print(f"\nMédia das consultas completamente julgadas: Precisão@{args.top_k}={sum(scores)/len(scores):.1%} ({len(scores)}/{len(cases)} consultas).")
    print(f"Adicione uma linha por candidato em {JUDGMENTS.name}: relevant=1 (relevante) ou 0 (não relevante).")
    print("Precisão@k é validação manual inicial; candidatos sem rótulo ficam pendentes, não irrelevantes.")
if __name__=="__main__": main()
