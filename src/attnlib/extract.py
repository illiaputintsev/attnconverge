"""Run models and keep their attention matrices."""

import os
import gc
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .metrics import conditional_content, exact_alignment, common_span_ends, aggregate_spans


MODEL_INFO = {
    "pythia-70m": {"hub": "EleutherAI/pythia-70m", "size_m": 70, "family": "Pythia", "organisation": "EleutherAI"},
    "pythia-160m": {"hub": "EleutherAI/pythia-160m", "size_m": 160, "family": "Pythia", "organisation": "EleutherAI"},
    "pythia-410m": {"hub": "EleutherAI/pythia-410m", "size_m": 410, "family": "Pythia", "organisation": "EleutherAI"},
    "pythia-1b": {"hub": "EleutherAI/pythia-1b", "size_m": 1000, "family": "Pythia", "organisation": "EleutherAI"},
    "gpt2": {"hub": "gpt2", "size_m": 124, "family": "GPT-2", "organisation": "OpenAI"},
    "gpt2-medium": {"hub": "gpt2-medium", "size_m": 355, "family": "GPT-2", "organisation": "OpenAI"},
    "gpt-neo-125m": {"hub": "EleutherAI/gpt-neo-125m", "size_m": 125, "family": "GPT-Neo", "organisation": "EleutherAI"},
    "opt-125m": {"hub": "facebook/opt-125m", "size_m": 125, "family": "OPT", "organisation": "Meta"},
    "crfm-gpt2-x21": {"hub": "stanford-crfm/alias-gpt2-small-x21", "size_m": 124, "family": "CRFM-GPT-2", "organisation": "Stanford CRFM"},
    "crfm-gpt2-x49": {"hub": "stanford-crfm/battlestar-gpt2-small-x49", "size_m": 124, "family": "CRFM-GPT-2", "organisation": "Stanford CRFM"},
}
EXTRACTION_VERSION = 2


SELECTION = "data/selection.txt"
EVALUATION = "data/evaluation.txt"
REPETITION = "data/repetition.txt"
N_TOKENS = 32
CACHE_PATH = "results/attention_cache"


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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def passages_hash(manifest):
    """Identify the ordered text and split labels independently of source paths."""
    return digest({k: manifest[k] for k in ("schema_version", "token_count", "records")})


def load_passages(selection_path, evaluation_path, repetition_path, tokens=32):
    """Read one excerpt per line; evaluation and repetition pair by line number."""
    groups = []
    for path in (selection_path, evaluation_path, repetition_path):
        # Retain spaces: stripping them can change token boundaries.
        lines = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError(f"Expected one excerpt per line, with no blank lines: {path}")
        if len(set(lines)) != len(lines):
            raise ValueError(f"Duplicate excerpts in {path}")
        groups.append(lines)
    selection, evaluation, repetition = groups
    if len(evaluation) != len(repetition):
        raise ValueError("Evaluation and repetition files must have the same number of lines")
    if set(selection) & (set(evaluation) | set(repetition)):
        raise ValueError("Selection/evaluation text overlap")
    if any(a == b for a, b in zip(evaluation, repetition)):
        raise ValueError("Each repetition must differ from its ordinary excerpt")

    records = [{"id": f"selection-{i:03d}", "split": "selection",
                "condition": "ordinary", "paired_id": None, "text": text}
               for i, text in enumerate(selection, 1)]
    for i, (ordinary, repeated) in enumerate(zip(evaluation, repetition), 1):
        pair = f"evaluation-{i:03d}"
        for condition, text in (("ordinary", ordinary), ("repetition", repeated)):
            records.append({"id": f"{pair}-{condition}", "split": "evaluation",
                            "condition": condition, "paired_id": pair, "text": text})
    return {"schema_version": 1, "token_count": tokens, "records": records}


def cache_dir(root, manifest, name):
    return Path(root).expanduser() / passages_hash(manifest)[:20] / name


def content_encoding(tokenizer, text):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [list(x) for x in enc["offset_mapping"]]
    native = tokenizer(text, add_special_tokens=True)["input_ids"]
    n_prefix = len(native) - len(ids)
    if n_prefix < 0 or list(native[n_prefix:]) != ids:
        raise ValueError("Only leading model-default special tokens are supported")
    return {"ids": ids, "offsets": offsets,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "native_ids": list(native), "prefix_ids": list(native[:n_prefix])}


def extract_passages(name, manifest, root, device="cpu", local_only=False,
                  model_cache=None, threads=2, revision=None):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if name not in MODEL_INFO:
        raise ValueError("Unknown model: " + name)
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    hub = MODEL_INFO[name]["hub"]
    kwargs = {"local_files_only": local_only}
    if model_cache:
        kwargs["cache_dir"] = model_cache
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(hub, use_fast=True, **kwargs)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with text offsets is required")
    model = AutoModelForCausalLM.from_pretrained(hub, attn_implementation="eager", dtype=torch.float32, **kwargs)
    model.eval().to(device)
    fingerprint = {"version": EXTRACTION_VERSION, "manifest_sha256": digest(manifest),
                   "hub": hub, "revision": getattr(model.config, "_commit_hash", None),
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   "device": device, "dtype": str(next(model.parameters()).dtype),
                   "attention_cache_dtype": "float16",
                   "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    out = cache_dir(root, manifest, name)
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        cached = previous["fingerprint"]
        # A provenance-only edit does not require another model forward pass.
        if (previous.get("passages_sha256") == passages_hash(manifest)
                and {k: v for k, v in cached.items() if k != "manifest_sha256"}
                == {k: v for k, v in fingerprint.items() if k != "manifest_sha256"}):
            fingerprint = cached
    fingerprint_hash = digest(fingerprint)
    metadata = {"fingerprint": fingerprint, "model": name, "properties": MODEL_INFO[name],
                "passages_sha256": passages_hash(manifest),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "representation": "final content token; HF hidden_states[1:] (final entry can include final layer norm)",
                "records": {}}
    try:
        for i, record in enumerate(manifest["records"]):
            enc = content_encoding(tokenizer, record["text"])
            if len(enc["ids"]) < 4:
                raise ValueError("Too few content tokens: " + record["id"])
            if len(enc["native_ids"]) > model.config.max_position_embeddings:
                raise ValueError("Passage exceeds model context: " + record["id"])
            target = out / (record["id"] + ".npz")
            reused = False
            if target.exists():
                with np.load(target, allow_pickle=False) as old:
                    reused = str(old["fingerprint"].item()) == fingerprint_hash
            if not reused:
                inputs = torch.tensor([enc["native_ids"]], device=device)
                p = len(enc["prefix_ids"])
                with torch.inference_mode():
                    res = model(input_ids=inputs, output_attentions=True,
                                output_hidden_states=True, use_cache=False)
                if not res.attentions or any(a is None for a in res.attentions):
                    raise RuntimeError("Model did not return eager attention")
                raw = np.stack([a[0].float().cpu().numpy() for a in res.attentions])
                # Sink means exclude the first content query in every model.
                queries = raw[:, :, p + 1:, :]
                bos = queries[:, :, :, :p].sum(axis=-1).mean(axis=-1)
                first = queries[:, :, :, p].mean(axis=-1)
                bos_per_query = raw[:, :, p:, :p].sum(axis=-1)
                first_per_query = raw[:, :, p:, p]
                hidden = np.stack([h[0, -1].float().cpu().numpy() for h in res.hidden_states[1:]])
                if hidden.shape[0] != raw.shape[0] or not np.isfinite(raw).all() or not np.isfinite(hidden).all():
                    raise RuntimeError("Invalid attention/hidden-state output")
                temporary = target.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, attention=raw[:, :, p:, p:].astype(np.float16),
                                    hidden=hidden, sink_bos=bos, sink_first_content=first,
                                    sink_bos_per_query=bos_per_query, sink_first_content_per_query=first_per_query,
                                    sink_start=bos + first, fingerprint=np.array(fingerprint_hash))
                temporary.replace(target)
                del res, raw, queries, hidden, bos_per_query, first_per_query
            metadata["records"][record["id"]] = {
                "text_sha256": digest(record["text"]), "offsets": enc["offsets"],
                "tokens": enc["tokens"], "prefix_ids": enc["prefix_ids"], "n_tokens": len(enc["ids"])}
            if (i + 1) % 25 == 0 or i + 1 == len(manifest["records"]):
                print(f"{name}: {i + 1}/{len(manifest['records'])} passages", flush=True)
        temp = out / "metadata.tmp.json"
        temp.write_text(json.dumps(metadata, indent=2) + "\n")
        temp.replace(out / "metadata.json")
    finally:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


class AttentionCache:
    def __init__(self, manifest, cache_root, names):
        self.manifest = manifest
        self.records = {r["id"]: r for r in manifest["records"]}
        self.names = list(names)
        self.paths = {name: cache_dir(cache_root, manifest, name) for name in names}
        self.metadata = {}
        for name, path in self.paths.items():
            metadata_path = path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing complete cache for {name}: run python -m src.attnlib.extract first ({metadata_path})")
            m = json.loads(metadata_path.read_text())
            if m["fingerprint"]["version"] != EXTRACTION_VERSION or m["model"] != name or m["fingerprint"]["hub"] != MODEL_INFO[name]["hub"]:
                raise ValueError("Cache extraction version/model mismatch: " + name)
            if m.get("passages_sha256") != passages_hash(manifest):
                raise ValueError("Cache/passages mismatch: " + name)
            if set(m["records"]) != set(self.records):
                raise ValueError("Cache is missing sample IDs: " + name)
            self.metadata[name] = m

    def aligned_ids(self, names, mode="exact", n_spans=16):
        accepted, rejected, spans = [], [], {}
        for sample_id, row in self.records.items():
            entries = [self.metadata[n]["records"][sample_id] for n in names]
            if any(e["text_sha256"] != digest(row["text"]) for e in entries):
                raise ValueError("Text identity changed in cache: " + sample_id)
            if mode == "exact":
                ok = exact_alignment(entries, row["text"], self.manifest["token_count"])
            elif mode == "spans":
                ends = common_span_ends([e["offsets"] for e in entries], len(row["text"]), n_spans)
                ok = ends is not None
                if ok:
                    spans[sample_id] = ends
            else:
                raise ValueError("Unknown alignment mode: " + mode)
            (accepted if ok else rejected).append(sample_id)
        return accepted, rejected, spans

    def load_hidden(self, name, ids):
        """Read final-content-token hidden states as [passage, layer, feature]."""
        if not ids:
            raise ValueError("Cannot load an empty cohort")
        expected = digest(self.metadata[name]["fingerprint"])
        hidden = []
        for sample_id in ids:
            with np.load(self.paths[name] / (sample_id + ".npz"), allow_pickle=False) as z:
                if z["fingerprint"].item() != expected:
                    raise ValueError("Stale per-example cache: " + sample_id)
                hidden.append(z["hidden"].astype(np.float32))
        return np.stack(hidden)

    def load(self, name, ids, spans=None):
        attention, hidden, bos, first, native_first, start = [], [], [], [], [], []
        for sample_id in ids:
            with np.load(self.paths[name] / (sample_id + ".npz"), allow_pickle=False) as z:
                if z["fingerprint"].item() != digest(self.metadata[name]["fingerprint"]):
                    raise ValueError("Stale per-example cache: " + sample_id)
                raw = z["attention"].astype(np.float32)
                if spans is not None:
                    offsets = self.metadata[name]["records"][sample_id]["offsets"]
                    ends = [end for _, end in offsets]
                    query_ids = [ends.index(end) for end in spans[sample_id]][1:]
                    raw = aggregate_spans(raw, offsets, spans[sample_id])
                    b = z["sink_bos_per_query"][:, :, query_ids].mean(axis=-1)
                    f = raw[:, :, 1:, 0].mean(axis=-1)
                    bos.append(b)
                    first.append(f)
                    start.append(np.clip(b + f, 0, 1))
                    native_first.append(z["sink_first_content_per_query"][:, :, query_ids].mean(axis=-1))
                else:
                    bos.append(z["sink_bos"])
                    first.append(z["sink_first_content"])
                    start.append(z["sink_start"])
                    native_first.append(z["sink_first_content"])
                attention.append(conditional_content(raw).astype(np.float16))
                hidden.append(z["hidden"])
        if not ids:
            raise ValueError("Cannot load an empty cohort")
        return {"attention": np.stack(attention), "hidden": np.stack(hidden),
                "sink_bos": np.stack(bos), "sink_first_region": np.stack(first),
                "sink_native_first_content": np.stack(native_first),
                "sink_start": np.stack(start)}


def split_passages(store, accepted):
    """Split accepted passages, keeping ordinary/repeated pairs in the same order."""
    records = store.records
    selection = [i for i in accepted if records[i]["split"] == "selection" and records[i]["condition"] == "ordinary"]
    ordinary = [i for i in accepted if records[i]["split"] == "evaluation" and records[i]["condition"] == "ordinary"]
    repetitions = {records[i]["paired_id"]: i for i in accepted
                   if records[i]["split"] == "evaluation" and records[i]["condition"] == "repetition"}
    paired = [i for i in ordinary if records[i]["paired_id"] in repetitions]
    repeat = [repetitions[records[i]["paired_id"]] for i in paired]
    return selection, {"ordinary": ordinary, "ordinary_paired": paired, "repetition": repeat}


def main():
    passages = load_passages(SELECTION, EVALUATION, REPETITION, N_TOKENS)
    for name in MODEL_INFO:
        extract_passages(name, passages, CACHE_PATH)


if __name__ == "__main__":
    main()
