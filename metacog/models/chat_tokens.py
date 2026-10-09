"""Normalize chat-template output to token IDs across tokenizer versions."""

from collections.abc import Mapping
from numbers import Integral


def diagnostic_turn_ids(tokenizer, question: str, enable_thinking: bool | None = None) -> list[int]:
    try:
        template_options = ({"enable_thinking": enable_thinking}
                            if enable_thinking is not None else {})
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            **template_options,
        )
        if isinstance(result, Mapping):
            result = result["input_ids"]
        if isinstance(result, str):
            result = tokenizer.encode(result, add_special_tokens=False)
        elif hasattr(result, "tolist"):
            result = result.tolist()
        if isinstance(result, (list, tuple)) and len(result) == 1 and isinstance(result[0], (list, tuple)):
            result = result[0]
        if not isinstance(result, (list, tuple)) or not result or not all(
            isinstance(token, Integral) for token in result
        ):
            raise ValueError(f"Chat template returned non-integer token IDs: {type(result).__name__}")
        return [int(token) for token in result]
    except Exception as exc:
        raise ValueError("Tokenizer cannot render the diagnostic user turn as token IDs") from exc
