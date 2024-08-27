from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from typing import List
from deepraghub.data.document import Document
from deepraghub.utils.logger import logger

# https://qdrant.tech/documentation/quickstart/


class VectorStore:
    def __init__(self, collection_name: str, embedding_model, url: str):
        self.client = QdrantClient(url)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Check if collection exists, if not create it
        self._create_collection_if_not_exists()

        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embedding_model.model,
        )

    def _create_collection_if_not_exists(self):
        collections = self.client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            # For OpenAI embeddings, we need to specify the dimension manually
            if hasattr(self.embedding_model, 'dimension'):
                dimension = self.embedding_model.dimension
            else:
                # Default dimension for OpenAI embeddings
                dimension = 1536  # This is the dimension for text-embedding-ada-002

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
            logger.info(f"Created new collection: {self.collection_name}")
        else:
            logger.info(f"Using existing collection: {self.collection_name}")

    def add_documents(self, documents: List[Document]):
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.vector_store.add_texts(texts, metadatas)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        results = self.vector_store.similarity_search(query, k=k)
        return [
            Document(content=result.page_content, metadata=result.metadata)
            for result in results
        ]
