import argparse
import json
import os
import sys

from llama_index.llms.openai import OpenAI

from chunking import split_texts_into_chunks
from pdf_extraction import extract_text_from_folder
from retrieval import load_vector_store_index, create_vector_store_index

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small RAG Demo")
    parser.add_argument("--create", action="store_true", help="Create a vector store for answering questions")
    parser.add_argument("--load", action="store_true", help="Load existing vector store for answering questions")
    parser.add_argument("--path", type=str, help="Path where to load or save the vector store",
                        required=True)

    args = parser.parse_args()

    if args.create:
        # Extract text from pdfs, split into chunks, create vector store and save it
        extracted_texts = extract_text_from_folder("data")
        extracted_texts_and_chunks = split_texts_into_chunks(extracted_texts)
        # Optional: Save intermediate step to json
        with open("data/extracted_data.json", "w", encoding="utf-8") as file:
            json.dump(extracted_texts_and_chunks, file)
        index = create_vector_store_index(persist_dir=args.path, chunks=extracted_texts_and_chunks)

    elif args.load:
        # Load existing index
        index = load_vector_store_index(persist_dir=args.path)

    else:
        print("Please specify either flag --load or flag --create! Exiting.")
        sys.exit(0)

    # Use loaded or created index to answer questions
    query_engine = index.as_query_engine(
        llm=OpenAI(model="gpt-4o"),
        system_prompt="Du bist ein hilfreicher Assistent. Antworte präzise und mit klaren Erklärungen. "
                      "Antworte immer auf deutsch.",
        # similarity_top_k=10,
        # response_mode="tree_summarize"
    )

    response1 = query_engine.query("Wie hoch ist die Grundzulage?")
    print(response1.response + "\n\n")

    response2 = query_engine.query("Wie werden Versorgungsleistungen aus einer Direktzusage oder einer "
                                   "Unterstützungskasse steuerlich behandelt?")
    print(response2.response + "\n\n")

    response3 = query_engine.query("Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem "
                                  "Pensionsfonds in der Auszahlungsphase besteuert?")
    # For this one, we add our sources to the response
    retrieved_chunks = []
    for node in response3.source_nodes:
        chunk_text = node.node.text
        chunk_metadata = node.node.metadata
        retrieved_chunks.append(f"Quelle: {chunk_metadata.get('source', 'Unknown')} "
                                f"(Chunk {chunk_metadata.get('chunk_index', 'N/A')}):\n\n{chunk_text}\n\n")

    # Format final response
    response_w_sources = f"Antwort:\n{response3.response}\n\nZitierte Quellen:\n" + "\n---\n".join(retrieved_chunks)
    print(response_w_sources)