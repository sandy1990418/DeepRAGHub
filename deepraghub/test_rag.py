import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepraghub.utils.config.config import load_config
from deepraghub.data.loader import load_documents
from deepraghub.data.preprocessor import chunk_documents
from deepraghub.embedding.model import EmbeddingModel
from deepraghub.retrieval.vector_store import VectorStore
from deepraghub.generation.llm import LLMModel
from deepraghub.rag.pipeline import RAGPipeline
from deepraghub.utils.logger import logger


def test_rag_pipeline():
    # Load configuration
    config = load_config("deepraghub/utils/config/config.yaml")

    # Load and preprocess documents
    raw_docs = load_documents(config.data.raw_dir)
    chunked_docs = chunk_documents(
        raw_docs,
        chunk_size=config.data.chunk_size,
        chunk_overlap=config.data.chunk_overlap,
    )

    # Initialize embedding model
    embedding_model = EmbeddingModel(config.embedding)

    # Embed documents
    embedded_docs = embedding_model.embed_documents(chunked_docs)

    # Initialize vector store and index documents
    vector_store = VectorStore(
        config.retrieval.collection_name,
        embedding_model,
        url=config.retrieval.vector_db_url,
    )
    vector_store.add_documents(embedded_docs)

    # Initialize LLM model
    llm_model = LLMModel(
        config.generation.model_type,
        config.generation.model_name,
        max_tokens=config.generation.max_tokens,
        temperature=config.generation.temperature,
    )

    # Create RAG pipeline
    rag_pipeline = RAGPipeline(
        llm_model,
        vector_store,
        max_context_size=config.rag.max_context_size,
        top_k_docs=config.rag.top_k_docs,
    )

    # Test query
    query = "What is the main difference between CNNs and RNNs?"
    answer = rag_pipeline.query(query)

    logger.info(f"Query: {query}")
    logger.info(f"Answer: {answer}")

    # You can add assertions here to check the quality of the answer
    assert len(answer) > 0, "The answer should not be empty"
    assert (
        "CNN" in answer and "RNN" in answer
    ), "The answer should mention both CNN and RNN"


if __name__ == "__main__":
    test_rag_pipeline()
