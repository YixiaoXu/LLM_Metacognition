import argparse
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .capture import DecoderLayerCapture, get_decoder_layers, last_token_logits_kwargs


def parse_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def parse_max_memory(items: Optional[List[str]]) -> Optional[Dict[int, str]]:
    if not items:
        return None
    result = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --max-memory item '{item}'. Use format GPU:MEM, e.g. 0:36GiB.")
        device_text, memory = item.split(":", 1)
        result[int(device_text)] = memory
    return result


def resolve_device_map(name: str):
    if name == "single":
        return {"": 0}
    if name == "none":
        return None
    return name


def model_family(model_path: str) -> str:
    name = str(model_path or "").lower()
    if "llama-2" in name:
        return "llama2"
    if "llama-3" in name or "llama3" in name:
        return "llama3"
    if "gemma" in name:
        return "gemma"
    if "qwen" in name:
        return "qwen"
    return "unknown"


def load_tokenizer(args: argparse.Namespace, padding_side: str = "left"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=getattr(args, "trust_remote_code", False),
            use_fast=True,
        )
    except ValueError as exc:
        message = str(exc)
        if "GemmaTokenizer" in message:
            raise RuntimeError(
                "Failed to load Gemma tokenizer. Your transformers installation likely does not support Gemma yet. "
                "Upgrade inside the active environment, for example: "
                "pip install -U 'transformers>=4.44.0' sentencepiece accelerate"
            ) from exc
        raise
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = padding_side
    return tokenizer


def load_model(args: argparse.Namespace):
    quantization_config = None
    gptq_backend = getattr(args, "gptq_backend", None)
    if gptq_backend:
        config = AutoConfig.from_pretrained(
            args.model_path,
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )
        raw_quant_config = getattr(config, "quantization_config", None)
        if isinstance(raw_quant_config, dict) and str(raw_quant_config.get("quant_method", "")).lower() == "gptq":
            raw_quant_config = dict(raw_quant_config)
            raw_quant_config["backend"] = gptq_backend
            try:
                from transformers import GPTQConfig

                quantization_config = GPTQConfig.from_dict_optimum(raw_quant_config)
            except Exception:
                quantization_config = raw_quant_config
        else:
            print(
                f"[model_utils] --gptq-backend={gptq_backend} was requested, "
                "but the model config does not expose a GPTQ quantization_config; ignoring."
            )

    kwargs = {
        "device_map": resolve_device_map(getattr(args, "device_map", "auto")),
        "max_memory": parse_max_memory(getattr(args, "max_memory", None)),
        "torch_dtype": parse_dtype(getattr(args, "dtype", "auto")),
        "trust_remote_code": getattr(args, "trust_remote_code", False),
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    attn_impl = getattr(args, "attn_implementation", None)
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.eval()
    return model


def load_model_and_tokenizer(args: argparse.Namespace, padding_side: str = "left"):
    tokenizer = load_tokenizer(args, padding_side=padding_side)
    model = load_model(args)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def generation_eos_token_id(model, tokenizer):
    """Preserve model-specific turn terminators instead of forcing one EOS id."""
    eos_ids = []

    def add(value) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            return
        if token_id >= 0 and token_id not in eos_ids:
            eos_ids.append(token_id)

    generation_config = getattr(model, "generation_config", None)
    add(getattr(generation_config, "eos_token_id", None))
    add(getattr(getattr(model, "config", None), "eos_token_id", None))
    add(getattr(tokenizer, "eos_token_id", None))

    # Some chat models keep the end-of-turn token only in the tokenizer. These
    # tokens are safe generation terminators and prevent Llama-3/Qwen/Gemma
    # responses from running until max_new_tokens after completing an answer.
    added_vocab = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    for token in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>"):
        if token in added_vocab:
            add(added_vocab[token])

    if not eos_ids:
        return None
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids


def model_input_device(model) -> torch.device:
    try:
        embedding = model.get_input_embeddings()
        if embedding is not None and hasattr(embedding, "weight"):
            device = embedding.weight.device
            if device.type != "meta":
                return torch.device(device)
    except Exception:
        pass
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for value in model.hf_device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value not in {"cpu", "disk"}:
                return torch.device(value)
    if hasattr(model, "device"):
        device = torch.device(model.device)
        if device.type != "meta":
            return device
    return next(model.parameters()).device


def num_hidden_layers_from_config(config) -> int:
    for name in ("num_hidden_layers", "n_layer", "num_layers", "decoder_layers"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise AttributeError("Cannot infer number of decoder layers from model config.")


def prompt_content(row: Dict, prompt_style: str) -> str:
    question = str(row.get("question") or row.get("prompt") or "").strip()
    if prompt_style == "data":
        return str(row.get("prompt") or question)
    if prompt_style == "direct":
        return (
            "Solve the following problem. Give only the final answer in the form #### <answer>.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    if prompt_style == "cot":
        return (
            "Solve the following problem step by step. At the end, write the final answer in the form #### <answer>.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def _chat_messages(content: str, args: argparse.Namespace, merge_system: bool = False) -> List[Dict[str, str]]:
    system_prompt = getattr(args, "system_prompt", "")
    if system_prompt and not merge_system:
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    if system_prompt and merge_system:
        content = f"{system_prompt.strip()}\n\n{content}"
    return [{"role": "user", "content": content}]


def _thinking_value(args: argparse.Namespace):
    value = getattr(args, "chat_template_enable_thinking", "auto")
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]], args: argparse.Namespace) -> Optional[str]:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    thinking = _thinking_value(args)
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return None


def _looks_like_broken_chat_template(rendered: str) -> bool:
    # Some local Llama-2 tokenizer configs expose a malformed Jinja template
    # that leaks fragments such as " == 'user')" into the prompt. Treat that as
    # unusable and fall back to our explicit family template.
    broken_fragments = (
        " == 'user')",
        " == \"user\")",
        "message['role']",
        'message["role"]',
        "{% if",
        "{% endif",
    )
    return any(fragment in rendered for fragment in broken_fragments)


def _fallback_chat_prompt(content: str, args: argparse.Namespace) -> str:
    family = model_family(getattr(args, "model_path", ""))
    system_prompt = getattr(args, "system_prompt", "")
    if system_prompt:
        content_with_system = f"{system_prompt.strip()}\n\n{content}"
    else:
        content_with_system = content
    if family == "llama2":
        if system_prompt:
            return f"<s>[INST] <<SYS>>\n{system_prompt.strip()}\n<</SYS>>\n\n{content} [/INST]"
        return f"<s>[INST] {content} [/INST]"
    if family == "llama3":
        pieces = ["<|begin_of_text|>"]
        if system_prompt:
            pieces.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt.strip()}<|eot_id|>")
        pieces.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
        pieces.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(pieces)
    if family == "gemma":
        return f"<bos><start_of_turn>user\n{content_with_system}<end_of_turn>\n<start_of_turn>model\n"
    if family == "qwen":
        pieces = []
        if system_prompt:
            pieces.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n")
        pieces.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
        return "".join(pieces)
    raise ValueError("Tokenizer has no usable chat template and no fallback template is known for this model.")


def build_generation_prompt(row: Dict, tokenizer, args: argparse.Namespace) -> str:
    content = prompt_content(row, getattr(args, "prompt_style", "data"))
    if not getattr(args, "use_chat_template", False):
        return content + str(row.get("assistant_prefix") or "")
    rendered = _apply_chat_template(tokenizer, _chat_messages(content, args), args)
    if rendered is None and getattr(args, "system_prompt", ""):
        rendered = _apply_chat_template(tokenizer, _chat_messages(content, args, merge_system=True), args)
    if rendered is None or _looks_like_broken_chat_template(rendered):
        rendered = _fallback_chat_prompt(content, args)
    # A generated-step semantic reference represents the same conversational
    # state under another tokenizer. Keep the raw prompt in the user turn, then
    # append the target model's decoded prefix after the assistant header.
    return rendered + str(row.get("assistant_prefix") or "")


def add_common_model_args(parser: argparse.ArgumentParser, include_generation_prompt: bool = False) -> None:
    parser.add_argument("--device-map", default="auto", choices=["auto", "balanced", "balanced_low_0", "sequential", "single", "none"])
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-memory", nargs="*", default=None)
    parser.add_argument("--attn-implementation", default=None, help="Optional transformers attention implementation, e.g. eager, sdpa, flash_attention_2.")
    parser.add_argument("--gptq-backend", default=None, help="Override GPTQ backend, e.g. gptq_triton or gptq_torch, to avoid incompatible auto-selected kernels.")
    parser.add_argument("--trust-remote-code", action="store_true")
    if include_generation_prompt:
        parser.add_argument("--prompt-style", choices=["data", "direct", "cot"], default="data")
        parser.add_argument("--use-chat-template", action="store_true")
        parser.add_argument("--system-prompt", default="")
        parser.add_argument("--chat-template-enable-thinking", choices=["auto", "true", "false"], default="auto")
