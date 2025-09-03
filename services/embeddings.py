
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_vectorstore(persist_directory: str):
    """
    Load existing Chroma vectorstore, or create a new empty one.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"}
    )

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        return Chroma.from_documents(
            documents=[],  # start empty
            embedding=embeddings,
            persist_directory=persist_directory
        )

def add_to_vectorstore(chunks, persist_directory: str, file_hash: str):
    """
    Add chunks from a PDF into the vectorstore.
    Uses GPU embeddings if available.
    """
    vs = get_vectorstore(persist_directory)

    # Assign unique IDs per chunk (to avoid duplicates of same file)
    ids = [f"{file_hash}_{i}" for i, _ in enumerate(chunks)]

    # Bulk insert
    vs.add_documents(chunks, ids=ids)
    return vs


# def deduplicate_chunks(chunks):
#     """
#     Remove exact duplicate chunks before storing or retrieving.
#     Useful for minimizing repeated content in answers.
#     """
#     unique_chunks = []
#     seen_texts = set()
#     for chunk in chunks:
#         text = chunk.page_content.strip()
#         if text not in seen_texts:
#             unique_chunks.append(chunk)
#             seen_texts.add(text)
#     return unique_chunks
