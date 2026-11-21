"""Load the local Hugging Face chat model once and wrap it for LangChain."""

from __future__ import annotations

from functools import lru_cache

from .config import CONFIG


@lru_cache(maxsize=1)
def load_llm(model_name: str = CONFIG.llm_model, load_in_4bit: bool = False):
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=CONFIG.max_new_tokens,
        do_sample=True,  # required for temperature/top_p to take effect
        temperature=CONFIG.temperature,
        top_p=CONFIG.top_p,
        return_full_text=False,  # return only the answer, not the echoed prompt
    )
    # ChatHuggingFace applies the model's own chat template (Zephyr's <|system|>/<|user|> format).
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), tokenizer=tokenizer)
