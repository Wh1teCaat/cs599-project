from pathlib import Path

import yaml

from retriever import RAG, RunMode


def load_paths():
    root = Path(__file__).resolve().parent
    config_path = root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = str(root / config["loader"]["data_path"])
    db_path = str(root / config["retriever"]["db_path"])
    cache_path = str(root / config["embedding"]["cache_path"])
    return data_path, db_path, cache_path


def main():
    data_path, db_path, cache_path = load_paths()

    query = "Grassou最害怕、一直纠结的那个念头具体是什么？"

    rag = RAG(data_path, db_path, cache_path, mode=RunMode.ONLINE)
    retriever = rag.get_retriever()

    docs = retriever.invoke(query)

    print(f"query: {query}")
    print(f"retrieved docs: {len(docs)}")

    for i, doc in enumerate(docs[:5], 1):
        print(f"\n--- doc {i} ---")
        print("content preview:")
        text = doc.page_content or ""
        print((text[:300] + "...") if len(text) > 300 else text)

        print("metadata:")
        for k, v in (doc.metadata or {}).items():
            if isinstance(v, str) and len(v) > 200:
                v = v[:200] + "..."
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
