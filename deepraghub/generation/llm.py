from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Union


class LLMFactory:
    @staticmethod
    def create_llm(
        model_type: str, model_name: str, **kwargs
    ) -> Union[OpenAI, ChatOpenAI, HuggingFacePipeline]:
        if model_type == "openai":
            if "gpt" in model_name.lower():
                return ChatOpenAI(model_name=model_name, **kwargs)
            return OpenAI(model_name=model_name, **kwargs)
        elif model_type == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            return HuggingFacePipeline(pipeline=pipe)
        elif model_type == "custom":
            # Implement custom model loading logic here
            raise NotImplementedError("Custom model loading not implemented yet")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class LLMModel:
    def __init__(self, model_type: str, model_name: str, **kwargs):
        self.llm = LLMFactory.create_llm(model_type, model_name, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        return self.llm(prompt, **kwargs)
