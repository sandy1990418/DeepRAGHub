# deepraghub/data/preprocessor.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deepraghub.data.document import Document


def chunk_documents(
    documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 50
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.content)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                Document(content=chunk, metadata={**doc.metadata, "chunk_id": i})
            )
    return chunked_docs
