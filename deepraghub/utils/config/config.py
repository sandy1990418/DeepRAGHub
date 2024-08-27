from pydantic import BaseModel, Field
from typing import Optional
import yaml


class DataConfig(BaseModel):
    raw_dir: str = Field(..., description="Directory containing raw documents")
    processed_dir: str = Field(
        ..., description="Directory to store processed documents"
    )
    chunk_size: int = Field(default=512, description="Size of document chunks")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")


class EmbeddingConfig(BaseModel):
    model_name: str = Field(..., description="Name of the embedding model")
    model_type: str = Field(
        ..., description="Type of the embedding model (openai, huggingface, custom)"
    )
    device: str = Field(default="cpu", description="Device to run the model on")


class RetrievalConfig(BaseModel):
    vector_db_name: str = Field(..., description="Name of the vector database")
    vector_db_url: str = Field(..., description="URL of the vector database")
    collection_name: str = Field(
        ..., description="Name of the collection in the vector database"
    )


class GenerationConfig(BaseModel):
    model_type: str = Field(
        ..., description="Type of the LLM (api, custom, or huggingface)"
    )
    model_name: str = Field(..., description="Name or path of the LLM")
    api_key: Optional[str] = Field(
        default=None, description="API key for API-based models"
    )
    max_tokens: int = Field(
        default=512, description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7, description="Temperature for text generation"
    )


class RAGConfig(BaseModel):
    max_context_size: int = Field(
        default=4096, description="Maximum context size for RAG"
    )
    top_k_docs: int = Field(
        default=5, description="Number of top documents to retrieve"
    )


class Config(BaseModel):
    data: DataConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    rag: RAGConfig


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
