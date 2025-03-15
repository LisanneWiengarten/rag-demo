from llama_index.core import Document
import json

import faiss

from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex


def load_llama_index():
    vector_store = FaissVectorStore.from_persist_dir("data/vector_store2")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="data/vector_store2"
    )
    index = load_index_from_storage(storage_context=storage_context)
    return index


def create_llama_index():
    persist_dir = "data/vector_store2"

    with open("data/chunk_data.json", "r") as file:
        pdf_data = json.load(file)

    documents = []
    for doc_id, doc_info in pdf_data.items():
        file_name = doc_info["file_name"]

        for idx, chunk in enumerate(doc_info["chunks"]):
            doc = Document(text=chunk, metadata={"source": file_name, "chunk_index": idx})
            documents.append(doc)

    faiss_index = faiss.IndexFlatL2(1536)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    index.storage_context.persist(persist_dir=persist_dir)
    index.set_index_id("legal_finance_pdfs")
    return index