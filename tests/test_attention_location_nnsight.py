"""Resolution of the nnsight attention-pattern location (`attention_location_pattern`).

The configured patterns address the attention-function call inside the attention module's source
as `attention_interface_2`. nnsight numbers every use of the name `attention_interface` in the
source, so the index of the call depends on the transformers version (with transformers 5 it is
`attention_interface_0`). These tests use tiny random models on CPU.
"""

import gc
from typing import Any

import pytest
import torch
from transformers import Gemma2Config, LlamaConfig

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import TopK

N_LAYERS = 2
TINY = dict(
    hidden_size=8,
    intermediate_size=16,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=4,
    num_hidden_layers=N_LAYERS,
    vocab_size=16,
    max_position_embeddings=64,
)
TOKENS = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])


@pytest.fixture(autouse=True)
def cleanup():
    yield
    gc.collect()


def _tiny_model(arch: str) -> Any:
    if arch == "llama":
        cfg = LlamaConfig(**TINY, architectures=["LlamaForCausalLM"])  # type: ignore
        hooks = {"feature_input_hook": "mlp.hook_in", "feature_output_hook": "mlp.hook_out"}
    else:
        cfg = Gemma2Config(**TINY, architectures=["Gemma2ForCausalLM"])  # type: ignore
        hooks = {"feature_input_hook": "hook_resid_mid", "feature_output_hook": "hook_mlp_out"}
    cfg._name_or_path = "openai-community/gpt2"  # only used to load a tokenizer
    transcoders = {
        layer: SingleLayerTranscoder(8, 32, TopK(4), layer, skip_connection=True)
        for layer in range(N_LAYERS)
    }
    model = ReplacementModel.from_config(
        cfg, TranscoderSet(transcoders, **hooks), backend="nnsight"
    )
    model.tokenizer.pad_token = model.tokenizer.eos_token  # type: ignore
    return model


def _attention_loc_paths(model) -> list[str]:
    paths = []  # filled in place: nnsight doesn't push unsaved locals out of the trace
    with model.trace(TOKENS):
        paths.extend(loc.path for loc in model.attention_locs)
    return paths


@pytest.mark.parametrize("arch", ["llama", "gemma2"])
def test_attention_locs_resolve(arch):
    model = _tiny_model(arch)
    assert "attention_interface_2" in model._attention_pattern

    paths = _attention_loc_paths(model)

    assert len(paths) == N_LAYERS
    for layer, path in enumerate(paths):
        assert f"layers.{layer}." in path
        assert path.endswith("nn_functional_dropout_0")


@pytest.mark.parametrize("arch", ["llama", "gemma2"])
def test_attention_locs_fall_back_to_existing_index(arch):
    """A configured index that doesn't exist resolves to the same op as the working pattern."""
    model = _tiny_model(arch)
    expected = _attention_loc_paths(model)

    model._attention_pattern = model._attention_pattern.replace(
        "attention_interface_2", "attention_interface_7"
    )

    assert _attention_loc_paths(model) == expected


def test_attention_locs_unrelated_error_is_raised():
    model = _tiny_model("llama")
    model._attention_pattern = "model.layers[{layer}].self_attn.source.no_such_op_0"

    with pytest.raises(AttributeError), model.trace(TOKENS):
        list(model.attention_locs)


def test_attribute_small_llama():
    """attribute() freezes the attention pattern through these locations."""
    model = _tiny_model("llama")
    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type: ignore
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type: ignore
        graph = attribute(TOKENS, model)
    finally:
        tokenizer_class.all_special_ids = original_all_special_ids  # type: ignore
    assert graph.adjacency_matrix.shape[0] == graph.adjacency_matrix.shape[1] > 0
