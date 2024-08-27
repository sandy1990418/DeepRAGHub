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


class TrainConfig(BaseModel):
    enabled: bool = Field(default=False, description="Whether training is enabled")
    dataset_path: Optional[str] = Field(
        default=None, description="Path to the training dataset"
    )
    output_path: Optional[str] = Field(
        default=None, description="Path to save the trained model"
    )
    epochs: int = Field(default=3, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size for training")
    eval_dataset_path: Optional[str] = Field(
        default=None, description="Path to the evaluation dataset"
    )
    eval_batch_size: int = Field(default=32, description="Batch size for evaluation")
    eval_steps: int = Field(default=100, description="Number of evaluation steps")
    save_steps: int = Field(
        default=100, description="Number of steps to save the model"
    )
    fp16: bool = Field(
        default=False, description="Whether to use 16-bit floating point precision"
    )
    push_to_hub: bool = Field(
        default=False, description="Whether to push the model to Hugging Face Hub"
    )
    hub_model_id: Optional[str] = Field(
        default=None, description="Model ID on Hugging Face Hub"
    )
    learning_rate: float = Field(default=2e-5, description="Learning rate for training")
    warmup_steps: int = Field(default=100, description="Number of warmup steps")
    weight_decay: float = Field(default=0.01, description="Weight decay for training")
    max_grad_norm: float = Field(default=1.0, description="Maximum gradient norm")
    max_steps: int = Field(
        default=10000, description="Maximum number of training steps"
    )
    logging_dir: Optional[str] = Field(
        default=None, description="Directory to store logs"
    )
    logging_steps: int = Field(default=10, description="Number of steps to log")
    save_total_limit: int = Field(
        default=None, description="Maximum number of checkpoints to save"
    )
    eval_strategy: str = Field(default="epoch", description="Evaluation strategy")
    eval_delay: int = Field(
        default=0, description="Number of steps to delay evaluation"
    )
    eval_accumulate_steps: int = Field(
        default=1, description="Number of steps to accumulate evaluation"
    )
    eval_on_train_dataset: bool = Field(
        default=False, description="Whether to evaluate on the training dataset"
    )
    eval_on_eval_dataset: bool = Field(
        default=True, description="Whether to evaluate on the evaluation dataset"
    )
    eval_on_predict_dataset: bool = Field(
        default=False, description="Whether to evaluate on the prediction dataset"
    )
    eval_on_predict_dataset_size: int = Field(
        default=1000,
        description="Number of samples to evaluate on the prediction dataset",
    )
    eval_on_predict_dataset_num_workers: int = Field(
        default=4, description="Number of workers to use for evaluation"
    )
    eval_on_predict_dataset_batch_size: int = Field(
        default=16, description="Batch size for evaluation on the prediction dataset"
    )
    eval_on_predict_dataset_num_workers: int = Field(
        default=4, description="Number of workers to use for evaluation"
    )
    eval_on_predict_dataset_batch_size: int = Field(
        default=16, description="Batch size for evaluation on the prediction dataset"
    )
    eval_step: int = Field(default=100, description="Number of steps to evaluate")


class EmbeddingConfig(BaseModel):
    model_type: str = Field(description="Type of embedding model")
    model_name: str = Field(description="Name of the embedding model")
    device: str = Field(default="cpu", description="Device to use for embedding")
    train: TrainConfig = Field(
        default_factory=TrainConfig, description="Training configuration"
    )
    custom_model_path: Optional[str] = Field(
        default=None, description="Path to custom model"
    )


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
