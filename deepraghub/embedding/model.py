import os
from typing import List
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer
from deepraghub.utils.logger import logger
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from sentence_transformers.evaluation import SimilarityFunction


class EmbeddingModel:
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()
        if self.config.train.enabled:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def _load_model(self):
        if self.config.model_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs={"device": self.config.device},
            )
        elif self.config.model_type == "openai":
            return OpenAIEmbeddings()
        elif self.config.model_type == "custom":
            return SentenceTransformer(
                self.config.model_name, device=self.config.device
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def train(self):
        if not self.config.train.enabled:
            logger.info("Training is not enabled in the configuration.")
            return

        if not isinstance(self.model, SentenceTransformer):
            logger.info("Converting model to SentenceTransformer for training.")
            self.model = SentenceTransformer(
                self.config.model_name, device=self.config.device
            )

        # Load datasets
        # train_dataset = load_dataset("sentence-transformers/stsb", split="train")
        # eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
        # test_dataset = load_dataset("sentence-transformers/stsb", split="test")

        # Load and prepare dataset
        dataset = self._load_dataset()
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        # Initialize loss function
        # train_loss = losses.CosineSimilarityLoss(model=self.model)
        # Define loss function with Matryoshka Representation
        train_loss = self._get_matryoshka_loss()

        # Initialize evaluator
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],  
            # Assume perfect similarity if no score is provided
            main_similarity=SimilarityFunction.COSINE,
            name="sts-dev",
        )

        # Define training arguments
        output_dir = os.path.join(
            self.config.train.output_path,
            f"{self.config.model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        # Prepare training arguments
        training_args = self._get_training_args()

        # Initialize trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            evaluator=evaluator,
            loss=train_loss,
        )

        trainer.train()

        logger.info(f"Model trained and saved to {output_dir}")

        # Evaluate on test set
        test_dataset = dataset["test"]
        if test_dataset is not None:
            test_evaluator = EmbeddingSimilarityEvaluator(
                sentences1=test_dataset["sentence1"],
                sentences2=test_dataset["sentence2"],
                scores=test_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name="sts-dev",
            )
            test_evaluator(self.model)
        else:
            logger.warning("No test dataset found for evaluation.")

        # Save the final model
        final_output_dir = os.path.join(self.config.train.output_path, "final")
        self.model.save(final_output_dir)

        # Push to Hub if configured
        if self.config.train.push_to_hub:
            try:
                self.model.push_to_hub(self.config.train.hub_model_id)
            except Exception as e:
                logger.error(f"Error uploading model to the Hugging Face Hub: {str(e)}")

        # # Update the current model to the trained one
        # self.model = HuggingFaceEmbeddings(
        #     model_name=final_output_dir,
        #     model_kwargs={"device": self.config.device},
        # )

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        if isinstance(self.model, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
            return self.model.embed_documents(documents)
        elif isinstance(self.model, SentenceTransformer):
            return self.model.encode(documents).tolist()
        else:
            raise ValueError("Unsupported model type for embedding documents")

    def embed_query(self, query: str) -> List[float]:
        if isinstance(self.model, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
            return self.model.embed_query(query)
        elif isinstance(self.model, SentenceTransformer):
            return self.model.encode(query).tolist()
        else:
            raise ValueError("Unsupported model type for embedding query")

    def _load_dataset(self):
        # Load dataset from Hugging Face or local file
        if self.config.train.dataset_source == "huggingface":
            dataset = load_dataset(self.config.train.dataset_name)
        else:
            dataset = load_dataset("json", data_files=self.config.train.dataset_path)

        # Preprocess dataset
        def preprocess_function(examples):
            return {
                "anchor": examples["sentence1"],
                "positive": examples["sentence2"],
            }

        return dataset.map(preprocess_function)

    def _get_matryoshka_loss(self):
        from sentence_transformers.losses import (
            MatryoshkaLoss,
            MultipleNegativesRankingLoss,
        )
        # ref: https://github.com/huggingface/blog/blob/main/zh/matryoshka.md
        # Adjust matryoshka dimensions to be less than or equal to the model's embedding dimension
        matryoshka_dimensions = [384, 256, 128, 64]  # Adjusted dimensions
        inner_loss = MultipleNegativesRankingLoss(self.model)
        return MatryoshkaLoss(
            self.model, inner_loss, matryoshka_dims=matryoshka_dimensions
        )

    def _get_training_args(self):
        return SentenceTransformerTrainingArguments(
            output_dir=self.config.train.output_path,
            num_train_epochs=self.config.train.num_epochs,
            per_device_train_batch_size=self.config.train.batch_size,
            learning_rate=self.config.train.learning_rate,
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_sts-dev_spearman_cosine",  # Updated metric
        )

    def save_model(self, path: str):
        self.model.save(path)

    @classmethod
    def load_model(cls, path: str, config):
        instance = cls(config)
        instance.model = SentenceTransformer(path)
        return instance


# TODO: the embedding model should be able to load a pre-trained model from the Hugging Face Hub
# TODO: Fine tune Reranker [https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker]
# TODO: using LoRA in embedding model
# TODO: Feature extraction from the embedding model BAAI/bge-large-zh-v1.5
