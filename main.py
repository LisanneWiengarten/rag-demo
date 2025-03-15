
from llama_index.llms.openai import OpenAI

from retrieval import load_llama_index

if __name__ == "__main__":
    # input_path = "data/pdfs"
    # pdfs_as_strings = extract_text_from_folder(input_path)

    # pdfs_as_chunks = split_pdfs_into_chunks(file_data)

    # index = create_llama_index()
    index = load_llama_index()

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