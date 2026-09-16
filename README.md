# attnconverge

Measuring agreement between attention patterns across Pythia, GPT-2, GPT-Neo, OPT and two Stanford CRFM GPT-2 reproductions, from 70M to 1B parameters. The experiments compare individual heads across model size, depth and input, motivated by the [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987).

## Why attention can be compared across models

Hidden states live in each model's own coordinate system. Pythia-70m uses 512 dimensions and 410m uses 1024, and even where the widths match, a coordinate has no guaranteed correspondence across models. An element-by-element comparison therefore has no shared basis.

Attention matrices are indexed by positions in the text. When token boundaries match, each cell describes attention between the same pieces of text in both models. Equal token counts alone are insufficient; E8 checks the boundaries explicitly. The quantity measured here is attention allocation. Agreement in the geometry of hidden representations is a separate comparison.

![Predominantly diagonal attention in Pythia-70m layer 0 head 3 and Pythia-160m layer 0 head 7](results/matched_pair.png)

Both heads attend mainly to the current token, despite occupying different head indices.

## Method

Each head in one model is compared with every eligible head at the corresponding relative depth in the other. Cosine similarity is averaged across the text set, keeping the best match for each source head. Head indices are not assumed to correspond.

The vectors contain causal entries, with the first-token column removed. This separates agreement over the remaining tokens from shared concentration on token 0. The zero upper triangle is omitted as redundant; including shared zeros would not change cosine similarity. Sink-heavy heads are filtered separately, with 0.9 as the main cutoff in E5–E8 and other cutoffs used for sensitivity checks.

E3–E7 select and evaluate matches on the same fifty sentences, comparing them with random head pairs drawn across layers. E7 adds two Stanford CRFM GPT-2 reproductions, giving fifteen pairs across four organisations. A positive gap in these experiments describes the selected matches on that text set.

E8 fixes the selected matches, random targets and sink exclusions on separate selection text. Random targets come from the same eligible heads at the corresponding layers. Evaluation compares each fixed pairing on the same held-out excerpt and on every different-excerpt pairing. Both matching directions are averaged; early and late cover the first and last quarters of relative depth. For OPT, BOS mass is recorded before its row and column are removed and the remaining attention is renormalised over content tokens.

## Results

Attention heads show correspondence across both model sizes and independently trained families. On held-out text, all twenty-one eligible pairs retain a positive late-layer gap over random heads, from +0.116 to +0.250. This includes six Pythia pairs and fifteen among GPT-2, GPT-2-medium, GPT-Neo, OPT and the two CRFM reproductions. The remaining twenty-four candidate pairs cross the two tokenisers, had too few selection excerpts with identical token boundaries, and were not scored.

The largest gap belongs to the two CRFM models, which share architecture, training data and recipe and differ only in random seed: +0.250 late, against +0.214 for the next highest pair. Agreement between two runs of one recipe is therefore a reference level that no cross-family pair reaches, and it is the comparison the earlier experiments had no way to make.

![Late-layer matched-minus-random and same-minus-different-text comparisons with bootstrap intervals](results/e8/late_contrasts.png)

Each point represents one model pair; bars show 95% paired-excerpt bootstrap intervals with the head matches held fixed.

Agreement generally weakens with depth, but late-layer correspondence remains present. Nineteen of the twenty-one held-out pairs have a smaller late than early gap; Pythia-410m against 1b instead rises from +0.175 to +0.209, and Pythia-70m against 1b from +0.114 to +0.142. In Pythia-70m and 160m, declining attention entropy accompanies increasing concentration on the first token. This shared concentration contributes to raw similarity, but correspondence also survives its removal: all twenty-one held-out late gaps remain positive at sink cutoffs 0.7, 0.9 and 1.0, although their magnitudes vary.

Relative depth provides an approximate layer alignment. On the fifty-sentence set, it identifies the best-matching layer for three of six layers between Pythia-70m and 160m, and five of twelve between GPT-2 and GPT-Neo. GPT-2's final layer matches GPT-Neo's second layer more closely than its final layer. Similar attention patterns can therefore occur at substantially different depths.

![Sink-cutoff sensitivity and layer alignment in Pythia-70m and 160m on the fifty-sentence set](results/e4_cutoff_and_grid.png)

The agreement contains a substantial input-independent component. For the six non-Pythia pairs of the original eight-model panel, the early gap over random heads averages +0.214 on the same text and +0.172 on different texts; the late averages are +0.141 and +0.035. Much of the early correspondence survives input replacement, while late correspondence is more input-sensitive. Across all twenty-one pairs, late matched similarity is +0.153 to +0.374 higher on the same text.

Size mismatch does not explain the depth profile on its own. Pairs of approximately equal size also decline, and the larger mean decline among unequal-sized pairs on the fifty-sentence set is confounded with GPT-2-medium appearing in every such pair. Among the six Pythia pairs, log size ratio correlates at −0.836 with early agreement. In the held-out comparison, late gap correlates at +0.401 with log smaller-model size, −0.347 with log size ratio and +0.212 with mean excluded-head share. On the twelve pairs of the original eight-model panel the same three correlations were +0.700, −0.086 and +0.453. None of them is stable to which pairs are included. These associations describe overlapping model pairs under different text sets, baselines and depth summaries; they do not isolate the effects of scale or sinks.

Repeating each excerpt's first eight-token prefix reduces the late gap by 0.016–0.063 in every scored pair. Head mappings remain fixed in this paired comparison; the input change combines repetition with altered content and lexical diversity.

Full measurements and experiment-specific interpretation are in [observations.md](observations.md); design choices are recorded in [decisions.md](decisions.md).

## Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pinned dependencies were checked with Python 3.9. For E8, extract the model outputs once, then run the comparison:

```bash
python -m src.attnlib.extract
python -m src.experiments.e8_heldout
```

The first extraction downloads the model weights. Later runs reuse outputs in `results/attention_cache/`. E8 prints the measurements and saves its summary matrix, uncertainty intervals, depth profiles and factor comparisons in `results/e8/`.

Earlier experiments run individually and create their own caches, for example:

```bash
python -m src.experiments.e7_families
```

E8 reads three plain-text files, with one excerpt per line:

- `data/selection.txt`: 100 excerpts used to choose head matches.
- `data/evaluation.txt`: 100 held-out excerpts used to test them.
- `data/repetition.txt`: 100 repeated variants, paired with evaluation excerpts by line number.

The supplied excerpts have 32 tokens under the Pythia tokeniser. Sources, preparation settings and licences are listed in [data/README.md](data/README.md).

To prepare your own text, set `SELECTION_SOURCE` and `EVALUATION_SOURCE` in `src/prepare_data.py` to separate text pools, one paragraph per line. Then run:

```bash
python -m src.prepare_data
python -m src.attnlib.extract
python -m src.experiments.e8_heldout
```

Preparation replaces the three input files. `N_SELECTION` and `N_EVALUATION` set the sample counts. For different file locations, update `SELECTION`, `EVALUATION` and `REPETITION` in preparation, extraction and E8; relative paths, absolute paths and `~/` paths work. Keep `N_TOKENS` consistent across these files if changing excerpt length.

Run the tests with:

```bash
python -m unittest discover -s tests -t .
```

## Next

Compare attention correspondence with hidden-state similarity, then extend the attention comparison across tokenisers using shared text spans.

## References

- Huh et al., [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987), 2024.
- Wu et al., [Similarity Analysis of Contextual Word Representation Models](https://arxiv.org/abs/2005.01172), 2020.
