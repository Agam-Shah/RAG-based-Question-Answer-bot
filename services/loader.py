# from langchain_community.document_loaders import PyPDFLoader

# def load_pdf(file_path: str):
#     loader = PyPDFLoader(file_path)
#     return loader.load()

from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_parallel(file_path: str, max_workers: int = 8):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()  # returns one Document per page

    # Parallelize page cleaning (if needed later)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        docs = list(executor.map(lambda p: p, pages))

    return docs
