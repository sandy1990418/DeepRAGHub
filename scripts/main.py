from deepraghub.utils.config import load_config
from deepraghub.data.loader import load_documents
from deepraghub.data.preprocessor import chunk_documents
from deepraghub.embedding.model import EmbeddingModel
from deepraghub.retrieval.vector_store import VectorStore
from deepraghub.generation.llm import LLMModel
from deepraghub.rag.pipeline import RAGPipeline
import deepraghub.utils.logger as logger


def main():
    # Load configuration
    config = load_config("config/config.yaml")

    # Load and preprocess documents
    logger.info("Loading and preprocessing documents...")
    raw_docs = load_documents(config.data.raw_dir)
    chunked_docs = chunk_documents(
        raw_docs,
        chunk_size=config.data.chunk_size,
        chunk_overlap=config.data.chunk_overlap,
    )

    # Initialize embedding model
    logger.info("Initializing embedding model...")
    embedding_model = EmbeddingModel(config.embedding)

    # Train embedding model if enabled
    if config.embedding.train.enabled:
        logger.info("Training embedding model...")
        embedding_model.train()

    # Embed documents
    logger.info("Embedding documents...")
    embedded_docs = embedding_model.embed_documents(chunked_docs)

    # Initialize vector store and index documents
    logger.info("Initializing vector store and indexing documents...")
    vector_store = VectorStore(
        config.retrieval.collection_name,
        embedding_model,
        url=config.retrieval.vector_db_url,
    )
    vector_store.add_documents(embedded_docs)

    # Initialize LLM model
    logger.info("Initializing LLM model...")
    llm_model = LLMModel(
        config.generation.model_type,
        config.generation.model_name,
        max_tokens=config.generation.max_tokens,
        temperature=config.generation.temperature,
    )

    # Create RAG pipeline
    logger.info("Creating RAG pipeline...")
    rag_pipeline = RAGPipeline(
        llm_model,
        vector_store,
        max_context_size=config.rag.max_context_size,
        top_k_docs=config.rag.top_k_docs,
    )

    # Example query
    question = "What is the difference between CNN and RNN?"
    logger.info(f"Querying: {question}")
    answer = rag_pipeline.query(question)
    logger.info(f"Answer: {answer}")

    # Query with sources
    logger.info("Querying with sources...")
    result_with_sources = rag_pipeline.query_with_sources(question)
    logger.info(f"Answer with sources: {result_with_sources}")


if __name__ == "__main__":
    main()
