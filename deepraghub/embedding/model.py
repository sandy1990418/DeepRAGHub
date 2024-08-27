from typing import List
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from deepraghub.data.document import Document
import json


class EmbeddingModel:
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        if self.config.model_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs={"device": self.config.device},
            )
        elif self.config.model_type == "openai":
            return OpenAIEmbeddings()
        elif self.config.model_type == "custom":
            if not self.config.custom_model_path:
                raise ValueError("Custom model path not specified")
            return HuggingFaceEmbeddings(
                model_name=self.config.custom_model_path,
                model_kwargs={"device": self.config.device},
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        texts = [doc.content for doc in documents]
        embeddings = self.model.embed_documents(texts)
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        return documents

    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_query(query)

    def train(self):
        if not self.config.train.enabled:
            print("Training is not enabled in the configuration.")
            return

        if (
            self.config.model_type != "huggingface"
            and self.config.model_type != "custom"
        ):
            raise ValueError(
                "Training is only supported for HuggingFace and custom models."
            )

        # Load training data
        train_examples = self._load_training_data()

        # Initialize the model for training
        model = SentenceTransformer(self.config.model_name)

        # Prepare the training dataloader
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=self.config.train.batch_size
        )
        train_loss = losses.CosineSimilarityLoss(model)

        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.train.epochs,
            warmup_steps=100,
            output_path=self.config.train.output_path,
        )

        print(f"Model trained and saved to {self.config.train.output_path}")

        # Update the current model to the trained one
        self.model = HuggingFaceEmbeddings(
            model_name=self.config.train.output_path,
            model_kwargs={"device": self.config.device},
        )

    def _load_training_data(self) -> List[InputExample]:
        train_examples = []
        with open(self.config.train.dataset_path, "r") as f:
            for line in f:
                data = json.loads(line)
                train_examples.append(
                    InputExample(
                        texts=[data["text1"], data["text2"]], label=data["similarity"]
                    )
                )
        return train_examples
