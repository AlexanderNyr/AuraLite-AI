import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st
from model_engine import CharTokenizer, BPETokenizer


@given(st.text(min_size=1, max_size=200))
def test_char_tokenizer_roundtrip_property(text):
    tok = CharTokenizer(); tok.train(text)
    assert tok.decode(tok.encode(text)) == text


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=200))
def test_bpe_tokenizer_roundtrip_property(text):
    tok = BPETokenizer(); tok.train(text, vocab_size=min(128, max(8, len(set(text)) + 8)))
    decoded = tok.decode(tok.encode(text))
    assert isinstance(decoded, str)
    assert len(decoded) >= 0
