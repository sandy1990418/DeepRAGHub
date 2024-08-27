import os
from typing import List
from datetime import datetime
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from deepraghub.utils.logger import logger


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
        train_dataset = load_dataset("sentence-transformers/stsb", split="train")
        eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
        test_dataset = load_dataset("sentence-transformers/stsb", split="test")

        # Initialize loss function
        train_loss = losses.CosineSimilarityLoss(model=self.model)

        # Initialize evaluator
        dev_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-dev",
        )

        # Define training arguments
        output_dir = os.path.join(
            self.config.train.output_path,
            f"{self.config.model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.train.epochs,
            per_device_train_batch_size=self.config.train.batch_size,
            per_device_eval_batch_size=self.config.train.batch_size,
            warmup_ratio=0.1,
            evaluation_strategy="steps",
            eval_steps=self.config.train.eval_steps,
            save_strategy="steps",
            save_steps=self.config.train.save_steps,
            save_total_limit=2,
            logging_steps=100,
            fp16=self.config.train.fp16,
            bf16=False,
            run_name="sts",
        )

        # Create and run trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=train_loss,
            evaluator=dev_evaluator,
        )

        trainer.train()

        logger.info(f"Model trained and saved to {output_dir}")

        # Evaluate on test dataset
        test_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=test_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-test",
        )
        test_evaluator(self.model)

        # Save the final model
        final_output_dir = f"{output_dir}/final"
        self.model.save(final_output_dir)

        # Push to Hub if configured
        if self.config.train.push_to_hub:
            try:
                self.model.push_to_hub(self.config.train.hub_model_id)
            except Exception as e:
                logger.error(f"Error uploading model to the Hugging Face Hub: {str(e)}")

        # Update the current model to the trained one
        self.model = HuggingFaceEmbeddings(
            model_name=final_output_dir,
            model_kwargs={"device": self.config.device},
        )

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
