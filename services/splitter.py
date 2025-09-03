# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from config.settings import TOKENIZER

# def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 20):
#     """
#     Token-aware chunking so flan-t5-base (512 tokens max) never overflows.
#     """
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=lambda text: len(TOKENIZER.encode(text, truncation=False)),
#     )
#     return splitter.split_documents(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import TOKENIZER

def split_documents(documents, chunk_size=500, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(TOKENIZER.encode(text, truncation=False)),
    )
    return splitter.split_documents(documents)
