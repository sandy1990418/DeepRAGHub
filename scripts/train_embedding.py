import os

from deepraghub.utils.config.config import load_config
from deepraghub.embedding.model import EmbeddingModel
import json


def create_training_data(input_file, output_file):
    with open(input_file, "r") as f:
        texts = f.readlines()

    training_data = []
    for i in range(len(texts) - 1):
        training_data.append(
            {
                "text1": texts[i].strip(),
                "text2": texts[i + 1].strip(),
                "similarity": 0.8,  # Assuming consecutive sentences are similar
            }
        )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")


def main():
    # Load configuration
    config = load_config("deepraghub/utils/config/config.yaml")

    # Create training data from raw documents
    create_training_data(
        "deepraghub/data/raw/cnn.txt", "deepraghub/data/processed/training_data.jsonl"
    )

    # Update config for training
    config.embedding.train.enabled = True
    # config.embedding.train.dataset_path = "deepraghub/data/processed/training_data.jsonl"
    # config.embedding.train.output_path = "deepraghub/data/embeddings/custom_model"
    config.embedding.model_type = "custom"
    config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Initialize embedding model
    embedding_model = EmbeddingModel(config.embedding)

    # Train the model
    embedding_model.train()

    print("Embedding model training completed.")


if __name__ == "__main__":
    main()
