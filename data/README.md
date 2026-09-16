# Text data

`sentences.txt` is the original sentence set for E1–E7.

E8 uses one excerpt per line in three files:

- `selection.txt`: 100 excerpts for selecting head matches.
- `evaluation.txt`: 100 held-out excerpts for evaluation.
- `repetition.txt`: 100 repetition variants. Line *n* corresponds to line *n* in `evaluation.txt`.

Keep the evaluation files in the same order, without blank lines. Spaces are preserved because they can affect tokenisation.

The supplied excerpts were sampled from separate train and test pools in [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext), configuration `wikitext-2-raw-v1`, retrieved on 11 September 2026. The source pools are included as `wikitext_train.txt` and `wikitext_test.txt`, with 600 dataset rows each.

The [source dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md) lists CC BY-SA 3.0 and GFDL. The supplied text is sampled from that dataset; the preparation below crops excerpts and creates repeated variants.

`python -m src.prepare_data` reproduces the three files with seed `20260911`. Each excerpt has 32 tokens under `EleutherAI/pythia-70m`, tokeniser revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42`. A repetition variant repeats the first eight-token prefix and is cut back to 32 tokens. This changes both repetition and lexical diversity.

Preparation removes blank lines, headings, duplicate source paragraphs, duplicate excerpts and overlap between the selection and evaluation pools. It checks token counts after slicing the raw text.
