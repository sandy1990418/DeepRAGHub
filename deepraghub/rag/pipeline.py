from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from deepraghub.retrieval.vector_store import VectorStore
from deepraghub.generation.llm import LLMModel
from typing import Dict, Any


class RAGPipeline:
    def __init__(
        self,
        llm_model: LLMModel,
        vector_store: VectorStore,
        max_context_size: int,
        top_k_docs: int,
    ):
        self.llm_model = llm_model
        self.vector_store = vector_store
        self.max_context_size = max_context_size
        self.top_k_docs = top_k_docs
        self.llm_chain = self._create_llm_chain()

    def _create_llm_chain(self):
        prompt_template = """Use the following pieces of context to answer the question at the end. \
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        return LLMChain(llm=self.llm_model.llm, prompt=PROMPT)

    def query(self, question: str) -> str:
        docs = self.vector_store.similarity_search(question, k=self.top_k_docs)
        context = "\n\n".join([doc.content for doc in docs])
        return self.llm_chain.run(context=context, question=question)

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        docs = self.vector_store.similarity_search(question, k=self.top_k_docs)
        context = "\n\n".join([doc.content for doc in docs])
        answer = self.llm_chain.run(context=context, question=question)
        return {"query": question, "result": answer, "source_documents": docs}
