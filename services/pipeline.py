from typing import List, Dict, Any
from langchain.vectorstores.base import VectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from config.settings import TOKENIZER


# def _format_docs(docs) -> str:
#     return "\n\n".join(d.page_content for d in docs)


class RAGPipeline:
    """
    Production-style OOP RAG wrapper using LCEL under the hood.
    - Keeps retriever, prompt, LLM, and orchestration in one place.
    - ask() returns answer + source docs for UI.
    """

    def __init__(self, vectorstore: VectorStore, llm, k: int = 4, custom_prompt: str = None):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.llm = llm

        template = custom_prompt or """
            You are a helpful assistant. Use the provided context to answer the question.
            Do not repeat sentences. Be concise.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

        self.prompt = ChatPromptTemplate.from_template(template)

        # build once: prompt -> llm -> string
        self.gen_chain = self.prompt | self.llm | StrOutputParser()

    def _build_context(self, unique_docs, max_tokens=256):
        context = ""
        current_tokens = 0
        for d in unique_docs:
            chunk_tokens = len(TOKENIZER.encode(d.page_content, truncation=False))
            if current_tokens + chunk_tokens > max_tokens:
                break
            context += "\n\n" + d.page_content
            current_tokens += chunk_tokens
        return context.strip()

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Returns {'answer': str, 'source_documents': List[Document]}
        Ensures deduplication of both context text and source documents.
        """
        docs = self.retriever.invoke(question)

        # Deduplicate by content before formatting
        seen_texts = set()
        unique_docs = []
        for d in docs:
            text = d.page_content.strip()
            if text not in seen_texts:
                unique_docs.append(d)
                seen_texts.add(text)

        # Build context from only unique docs
        context = self._build_context(unique_docs, max_tokens=256)

        # Generate answer
        answer = self.gen_chain.invoke({"context": context, "question": question})

        return {"answer": answer, "source_documents": unique_docs}


    # def batch(self, questions: List[str]) -> List[Dict[str, Any]]:
    #     return [self.ask(q) for q in questions]
