# Decisions

## 1. What this project measures

The Platonic Representation Hypothesis (Huh et al., 2024) measures whether models converge on their **representations** (the geometry of what they know internally). Our project measures whether they converge on their **attention patterns** (how each head allocates its softmax-normalised weight across the preceding tokens).

Attention weights are produced by separately learned query and key matrices sitting on top of the representations, so two models can hold similar knowledge and still have learned different lookup strategies over it. Wu et al. (2020) compared both signals across models and found they can disagree, which is direct evidence that one does not imply the other.

The object of measurement is the attention patterns themselves, not the query and key projections that produce them. Those projections are parameterised by each model's hidden dimension, which means they inherit the incomparability of the hidden states. The patterns avoid this, because both models produce an n × n matrix indexed by the same tokens.

## 2. Model choice

All Pythia sizes share one tokeniser, so the same sentence produces the same tokens in every model. That makes the attention matrices identical in shape with identical meaning per cell, so they can be compared directly with no alignment machinery. The difference in values is the main scope of this research project.

They also share training data and architecture recipe, so scale is the only variable. Rising agreement would extend PRH's convergence claim from representations to attention. If agreement does not rise with scale, the two forms of convergence are separable. Models may arrive at similar internal geometry while retaining distinct patterns of attention. Attention would then need to be measured in its own right rather than inferred from representational similarity.

## 3. Other models

Different tokenisers segment the same sentence differently, so the two matrices are built over mismatched indices and their cells describe different pairs of text spans. Comparing across families needs sub-token merging to produce word-level matrices, which is a design problem of its own.

The later steps of the project should include comparisons like GPT-2 versus GPT-Neo. These models share a tokeniser with each other, so the same pipeline would work without sub-token merging, and it would test convergence between models built independently by different organisations.