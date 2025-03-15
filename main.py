import argparse
import json
import os
import sys

from llama_index.llms.openai import OpenAI
from chunking import split_texts_into_chunks
from pdf_extraction import extract_text_from_folder
from retrieval import load_vector_store_index, create_vector_store_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small RAG Demo")
    parser.add_argument("--create", action="store_true", help="Create a vector store for answering questions")
    parser.add_argument("--load", action="store_true", help="Load existing vector store for answering questions")
    parser.add_argument("--path", type=str, help="Path where to load or save the vector store",
                        required=True)

    args = parser.parse_args()

    if args.create:
        # Extract text from pdfs, split into chunks, create vector store and save it
        extracted_texts = extract_text_from_folder("data/pdfs")
        extracted_texts_and_chunks = split_texts_into_chunks(extracted_texts)
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
                      "Falls möglich, nenne Quellen aus den Dokumenten. Antworte immer auf deutsch.")

    response1 = query_engine.query("Wie hoch ist die Grundzulage?")
    print(response1.response)

    response2 = query_engine.query("Wie werden Versorgungsleistungen aus einer Direktzusage oder einer "
                                   "Unterstützungskasse steuerlich behandelt?")
    print(response2.response)

    response3 = query_engine.query("Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem "
                                  "Pensionsfonds in der Auszahlungsphase besteuert?")
    print(response3.response)