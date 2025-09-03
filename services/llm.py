from transformers import AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
from config.settings import MODEL_NAME, TOKENIZER

def build_llm(model_name: str = MODEL_NAME) -> HuggingFacePipeline:
    """
    Build a local Seq2Seq LLM (T5/BART) as a LangChain Runnable.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    gen = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=TOKENIZER,
    max_new_tokens=256,
    truncation=True,  # optional safety
)

    # Pass generation parameters via HuggingFacePipeline
    return HuggingFacePipeline(pipeline=gen, 
                               model_kwargs={"temperature": 0.7,
                                            "top_p": 0.9,
                                            "repetition_penalty": 1.2,
                                            }
                                )
