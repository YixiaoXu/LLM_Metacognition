"""Chat-template outputs must be integer IDs before model forward."""

import pytest

from metacog.models.chat_tokens import diagnostic_turn_ids


class TemplateTokenizer:
    def __init__(self, result):
        self.result = result
        self.encoded = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        return self.result

    def encode(self, rendered, add_special_tokens):
        self.encoded = (rendered, add_special_tokens)
        return [1, 12, 13]


@pytest.mark.parametrize("result", ["<s>[INST] Question [/INST]", [1, 12, 13],
                                          {"input_ids": [1, 12, 13]}, [[1, 12, 13]]])
def test_diagnostic_turn_ids_normalizes_template_output(result):
    tokenizer = TemplateTokenizer(result)
    assert diagnostic_turn_ids(tokenizer, "Question") == [1, 12, 13]
    if isinstance(result, str):
        assert tokenizer.encoded == (result, False)


def test_diagnostic_turn_ids_rejects_text_tokens():
    with pytest.raises(ValueError, match="token IDs"):
        diagnostic_turn_ids(TemplateTokenizer(["<s>", "Question"]), "Question")
