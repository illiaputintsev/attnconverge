"""Run models and keep their attention matrices."""

import os
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_sentences(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def extract(name, sentences, min_tokens=4, half=True, verbose=True):
    """Run a sentence set through a model and keep every attention matrix.
    Returns (attn, n_layers, n_heads, tokens) where attn is indexed
    [sentence][layer] and each element is [heads, seq, seq]."""

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name,
                                                 attn_implementation="eager")
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    probe = tok("The animal was tired", return_tensors="pt")
    probe_toks = tok.convert_ids_to_tokens(probe["input_ids"][0])
    has_bos = probe_toks[0] in (tok.bos_token, "<s>", "</s>")

    if verbose:
        print(f"{name}: {n_layers} layers, {n_heads} heads"
              + ("(drops a leading BOS token)" if has_bos else ""))

    out, first_tokens = [], None
    for i, text in enumerate(sentences):
        inputs = tok(text, return_tensors="pt")
        toks = tok.convert_ids_to_tokens(inputs["input_ids"][0])
        if has_bos:
            toks = toks[1:]
        if len(toks) < min_tokens:
            continue
        if first_tokens is None:
            first_tokens = toks

        with torch.no_grad():
            res = model(**inputs, output_attentions=True)

        mats = []
        for layer in res.attentions:
            m = layer[0].numpy()
            if has_bos:
                m = m[:, 1:, 1:]
                s = m.sum(axis=-1, keepdims=True)
                m = np.divide(m, s, out=np.zeros_like(m), where=s > 0)
            mats.append(m.astype(np.float16) if half else m)
        out.append(mats)

        if verbose and (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(sentences)}")

    return out, n_layers, n_heads, first_tokens


def get_data(models, sentences, cache_path, **kwargs):
    """Extract model once, cache the result."""
    if os.path.exists(cache_path):
        print(f"loading cached attention from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"extracting attention for {len(models)} models "
          f"(first run, several minutes)")
    data = {}
    for name in models:
        data[name.split("/")[-1]] = extract(name, sentences, **kwargs)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"cached to {cache_path}")
    return data


def load_cache(path):
    """Read a cache written by an earlier experiment, keyed by short name."""
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {name.split("/")[-1]: entry for name, entry in d.items()}