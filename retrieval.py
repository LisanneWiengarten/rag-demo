import faiss
from llama_index.core import Document
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.faiss import FaissVectorStore


def load_vector_store_index(persist_dir: str) -> BaseIndex:
    """
    Loads a vector store index from the given directory
    :param persist_dir: Directory from where to load the index
    :return: The loaded index
    """
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
    index = load_index_from_storage(storage_context=storage_context)
    return index


def create_vector_store_index(persist_dir: str, chunks: dict) -> VectorStoreIndex:
    """
    Creates a vector store index using Llama Index and FAISS. Returns the index for immediate usage
    :param persist_dir: Where to store the created vector store index
    :param chunks: A dictionary with ids as keys, and dictionaries with file_name, extracted texts and chunks as values
    :return: A VectorStoreIndex based on the given documents, saved to the given directory
    """
    # Create input documents first
    documents = []
    for doc_id, doc_info in chunks.items():
        file_name = doc_info["file_name"]
        for idx, chunk in enumerate(doc_info["chunks"]):
            doc = Document(text=chunk, metadata={"source": file_name, "chunk_index": idx})
            documents.append(doc)

    # Init FAISS index and storage context
    faiss_index = faiss.IndexFlatL2(1536)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Init VectorStoreIndex based on given documents and storage context
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    index.storage_context.persist(persist_dir=persist_dir)
    return index