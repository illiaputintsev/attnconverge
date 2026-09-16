# Observations

## E1. One sentence, both models

A single 12-token sentence was run through 70m and 160m. Both returned matrices of the same shape over the same tokens, with every row summing to one and the upper triangle at zero.

Row entropy declines with depth in both models, beginning around 1.1 nats and reaching 0.11 for 70m and 0.02 for 160m. Exponentiated, this means each token draws on roughly three tokens' worth of attention in the early layers and approximately one by the end. A head distributing its weight evenly across twelve tokens would place about 0.26 on token 0, so the late layers exceed that by a factor of three.

The two measurements describe the same phenomenon from different angles. Since every row must sum to one, weight accumulating on token 0 leaves less available elsewhere. Entropy falling with depth could be read as the heads narrowing onto whichever token carries the relevant information, but the per-head weights show much of the concentration landing on token 0.

At the level of individual heads, both first layers contain one positioned almost entirely on the diagonal (70m H3, 160m H7), alongside heads placing their weight one position back and others distributing it across recent tokens with distance decay. The first layer can use token identity and position, but has no contextualised representations from earlier layers. Both final layers are dominated by sinks, though each retains a single stepped-diagonal head (70m H7, 160m H11).

## E2. Fifty sentences

The measurement was repeated across fifty sentences of comparable length, with entropy normalised so that rows of different lengths could be averaged together, and with baselines added to both plots.

The E1 shapes persisted across the set. Normalised entropy falls from 0.65 in 70m and 0.72 in 160m at the first layer to 0.09 and 0.02 at the last, while sink fraction rises from 0.22, the value an evenly distributed head would produce at these sentence lengths, to approximately 0.8. The second layer of 70m places only 0.18 on token 0, below the 0.22 an evenly distributed head produces, so the early layers are directing attention away from the first token rather than towards it.

Sink fraction varies substantially between heads inside a single layer. This is why the sink curve carries wide error bands from mid-depth onwards. In 70m layer 4, heads 2, 3 and 7 approach complete sinking while heads 0 and 5 remain near 0.5. In layer 5, heads 2 and 7 place almost no weight on token 0 while heads 3 and 6 place nearly all of it. In neither layer does the mean correspond to the behaviour of any individual head.

Both models exhibit the same structure: low sink fraction through the early layers, a sharp rise around layer 3 of 6 in 70m and layer 4 of 12 in 160m, and a high band extending to the end. Heads cluster near zero or near one rather than distributing evenly between them.

## E3. Cross-model head matching, 70m and 160m

Every head in 70m was compared against every head in 160m by cosine similarity over their attention matrices, averaged across the fifty sentences, keeping the best match for each head in 70m. Layers were paired by relative depth. Head numbering follows from initialisation and means nothing across models, so the comparison runs all against all rather than by index. Heads above 0.7 sink fraction were excluded.

The random-pair baseline is roughly 0.50 at every depth, between 0.499 and 0.526. Results are therefore given as the gap over that baseline rather than as raw similarity.

Best match reaches 0.824 at the first layer pair and 0.809 at the second, gaps of +0.325 and +0.294 over the baseline. The gap falls to +0.125 and +0.178 at the two middle pairs, then to −0.272 at depth 0.8 and −0.050 at the last layer.

At depth 0.8 the comparison involves two heads against three, and at depth 1.0 five against three, so the collapse cannot be attributed to the models rather than the filter on this evidence.

Keeping column 0 raised every figure. At depth 0.6 the masked comparison gave 0.690 and the unmasked 0.860. Both models placed most of their late weight on the first token, so the unmasked figures included shared concentration on token 0, which does not establish agreement in routing among the remaining tokens.

## E4. Filter sensitivity and layer alignment, 70m against 160m

The matching from E3 was repeated at four sink cutoffs: 0.7, 0.8, 0.9, and 1.0 which keeps every head. Every layer of 70m was then compared against every layer of 160m, and a within-model baseline was computed by comparing 70m heads against other 70m heads.

With no filter, where all eight heads are used at every depth, the gap runs +0.498, +0.484, +0.366, +0.237, −0.038, +0.099. The early-to-late decline appears at all four cutoffs, so it does not depend on the strict filter used in E3. The exact late gaps still vary with the cutoff.

Best match is identical at 0.824 in the first layer pair across all four cutoffs, while the mean baseline moves from 0.506 at cutoff 0.7 to 0.319 with no filter. Removing the filter therefore lowers the baseline while leaving the first-layer match unchanged. Once column 0 has been dropped, admitting sink-heavy heads lowers the mean similarity of random pairs.

Relative depth was correct for 3 of 6 layers. L0, L1 and L3 matched where predicted, L2 was off by one, and L4 and L5 matched L7 and L3 where relative depth predicted L9 and L11. Both scored 0.464 against a floor near 0.44, so neither matched anything well anywhere in the larger model. The six layers of 70m mapped onto four distinct layers of 160m, with L3 and L4 both matching L7.

Two random heads from 70m scored 0.459 against each other, and a random 70m head against a random 160m head scored 0.438, a difference of −0.020.

## E1-E4. Conclusions

Best-match similarity is high in the early layers and much weaker in the late ones. The gap reaches +0.325 at the first layer pair in E3 and turns negative in the final third. The decline persists at every sink cutoff, including none, although the final-layer gap is positive at some cutoffs.

Two random heads agree to roughly 0.44 whether they come from the same model or different ones, which is why the figures are reported as gaps over that floor.

Relative depth identifies the best match for three of the first four layers, with the remaining prediction off by one. The last two layers of 70m each reach a maximum of 0.464 across the larger model, against a floor near 0.44.

These results come from one pair of models trained on the same data by the same organisation, so the next step would be to test whether they hold across sizes and across models trained independently. Adding further Pythia sizes tests whether agreement grows with scale, as Platonic Representation Hypothesis claims for representations. 

## E5. Agreement across four Pythia sizes

Every pair of models was compared head by head, as in E3, and the gap over the random-pair baseline recorded at each relative depth. This gave six pairs, from 70m to 1b, at a 0.9 sink cutoff.

70m and 160m are 2.3x apart and reach a mean gap of +0.401 across the first two layer pairs. 410m and 1b are 2.4x apart, about six times larger, and reach +0.331. At these similar size ratios, the larger pair has lower early agreement. Agreement therefore does not increase with absolute scale in this comparison, although this does not establish that scale has no effect.

The correlation between log size ratio and early gap is −0.836 across the six pairs, and the pair furthest apart in size has the lowest early gap, at +0.260. Greater size separation is associated with lower early agreement in this set. The pairs share models, however, and scale changes depth and head configuration as well as parameter count, so the correlation does not isolate a size-ratio effect.

Relative depth identifies the best-matching layer for 3 of 6 layers between 70m and 160m, and 1 of 6 for both pairs putting 70m against a model above 400m. Matching layers by their fraction of the stack becomes less reliable in these wider size comparisons.

## E6. Independently trained models, GPT-2 against GPT-Neo

GPT-2 and GPT-Neo-125m were matched head by head on the same fifty sentences at a 0.9 sink cutoff. They share a tokeniser and are almost equal in size, at 124M and 125M parameters, but were trained separately by OpenAI and EleutherAI on WebText and the Pile.

Early agreement is +0.302 over the first two layer pairs, within the Pythia range from E5. The mean gap falls to +0.155 over the last three pairs, never drops below +0.100, and finishes at +0.178. Agreement weakens with depth, but the pair retains a positive gap through the end of the stack.

Relative depth identifies the best-matching layer for 5 of 12 layers. GPT-2 L11 matches GPT-Neo L1 at 0.808, compared with 0.723 against L11. Late-layer attention can therefore resemble much earlier attention in the other model, even when both models have the same number of layers.

The late collapse observed in Pythia is therefore not universal across the tested models. Whether size mismatch contributes remains open: every Pythia pair in E5 differed in size, whereas this pair does not. E7 adds unequal-sized comparisons outside Pythia.

## E7. Size mismatch across model families

GPT-2, GPT-2-medium, GPT-Neo-125m, OPT-125m and two Stanford CRFM GPT-2 reproductions were compared on the same fifty sentences at a 0.9 sink cutoff. The six models give fifteen pairs across four organisations. The CRFM models, alias-x21 and battlestar-x49, are GPT-2-small trained from scratch on OpenWebText and differ from each other only in random seed, so their pair measures what agreement survives when nothing but initialisation changes. For OPT, the leading BOS row and column were removed and the remaining rows renormalised, so its scores describe attention conditional on the sentence tokens.

All fifteen pairs have a lower late than early gap, but every late average remains positive, between +0.118 and +0.272. The seed-only pair is the highest at +0.272 and declines the least, by +0.078; the next highest late gap is +0.217 and the next smallest decline is +0.100. Same-recipe agreement is therefore a reference level that no cross-family pair reaches, and one the earlier experiments had no way to measure.

Ten pairs are approximately equal in size and five are 2.8–2.9x apart. The near-equal group has a mean decline of +0.123 and a late gap of +0.183, against +0.166 and +0.143 for the unequal group. Every unequal pair involves GPT-2-medium, however, so this split compares one model against the rest rather than size mismatch as such, and nothing about parameter count follows from it. Early and late mean the first two and last three source layers; these cover narrower depth intervals when the 24-layer GPT-2-medium is the source.

Early gaps across organisations range from +0.239 to +0.371, averaging +0.305 over thirteen pairs. Early correspondence is therefore present across each pairing of OpenAI, EleutherAI, Meta and Stanford CRFM models.

GPT-2 against GPT-Neo has an early gap of +0.321 here and +0.302 in E6. The best-match scores agree at the reported precision; the difference comes from the random-pair baseline being sampled again.

These gaps describe best matches selected and evaluated on the same sentences. The random-pair baseline does not account for choosing the best of several candidates, so a positive gap alone does not establish that the correspondence generalises to new text or depends on the input.

## E8. Held-out matching and input dependence

Head matches were selected on separate text at a 0.9 sink cutoff, then fixed for evaluation against random head pairs and different-text controls. Early and late mean the first and last quarters of relative depth, averaged across both matching directions. Twenty-one of the 45 pairs had enough 32-token excerpts with identical token boundaries: six Pythia pairs with 95 evaluation excerpts and fifteen among GPT-2, GPT-2-medium, GPT-Neo, OPT and the two CRFM reproductions with 54. The remaining twenty-four pairs cross the two tokenisers and had only sixteen aligned selection excerpts, below the minimum of 25.

Every scored pair retains a positive late gap over random heads, from +0.116 to +0.250, and higher matched similarity on the same text than on different texts, by +0.153 to +0.374. The correspondence therefore persists beyond the text used to select the heads. Nineteen pairs have a lower late than early gap; Pythia-410m against 1b rises from +0.175 to +0.209 and Pythia-70m against 1b from +0.114 to +0.142. Text, depth bands and averaging also differ from the earlier experiments, so changes in the depth profile cannot be attributed to held-out selection alone.

The seed-only pair, CRFM x21 against x49, has the highest late gap at +0.250 and the largest same-minus-different difference at +0.374. Two models sharing architecture, data and recipe, differing only in initialisation, therefore agree more than any pair drawn from different families, and by a clear margin: the next highest late gap is +0.214, for GPT-2 against CRFM x21.

For the six non-Pythia pairs of the original eight-model panel, the early matched-minus-random gap averages +0.214 on the same text and +0.172 on different texts; the late averages are +0.141 and +0.035. Much of the early correspondence survives replacing the input, while late correspondence is more input-sensitive. Across those twelve pairs, the late gap over random heads is +0.082 to +0.137 larger on the same text than on different texts. This difference accounts for random heads also benefiting from shared text. What information the matched heads follow remains untested. These two averages have not yet been recomputed over the nine pairs the CRFM models add.

Late gap correlates at +0.401 with log smaller-model size, −0.347 with log size ratio and +0.212 with mean excluded-head share. On the twelve pairs of the original eight-model panel the same three correlations were +0.700, −0.086 and +0.453, so none of them is stable to which pairs are included and none should be read as an effect of scale or sinks. The size association stays positive in both sets; it does not support scale independence. Per-model late means range from +0.137 to +0.193, over three partners for the Pythia models and five for the others. Every late gap also remains positive at sink cutoffs 0.7 and 1.0, with varying magnitudes.

Repeating each excerpt's first eight-token prefix reduces the late gap by 0.016–0.063 in every pair, using 95 paired excerpts for Pythia and 51 for the other fifteen pairs. For GPT-2 against GPT-Neo, the change is −0.032, with a paired 95% excerpt-bootstrap interval of [−0.039, −0.025] conditional on the fixed matches. The repetition condition also changes content and lexical diversity.

## E9. Attention agreement against representation agreement

Linear CKA was computed between the final-content-token hidden states of each pair, with its exact permutation expectation subtracted. That expectation is the value CKA takes when examples are paired at random, and removing it corrects for the finite-sample baseline. CKA compares the example-by-example similarity structure of the two sets of hidden states. Models of different width can therefore be compared without a shared basis across their dimensions. All 45 pairs were measured on the shared passage set. Twenty-one of them also have attention numbers on the exactly aligned subset, and only those 21 allow the two signals to be compared.

Late excess CKA runs +0.192 to +0.460 across the 45 pairs. The same-family and cross-family means are +0.465 and +0.437 early, and +0.368 and +0.322 late. Family membership separates representation agreement far less than it separates attention agreement.

Across the 21 comparable pairs, late attention gap and late aligned excess CKA correlate at +0.055. GPT-2 against CRFM x21 has the second-highest late attention gap at +0.214 and one of the lowest aligned excess CKA values at +0.202. GPT-2 against GPT-2-medium reverses both, at +0.160 and +0.386. Representational similarity therefore does not predict attention similarity in this set. Wu et al. (2020) reported the same separation between the two signals. The 21 pairs are few and share models; the correlation describes them and does not isolate a relationship between the two signals.

Late input excess and late aligned excess CKA correlate at −0.262 over the same pairs. What information the matched heads follow remains untested.

## E10. Attention correspondence across tokenisers

Each passage was divided into 16 spans on boundaries shared by all ten models, and attention was measured from the final token of each span, with key weights summed within spans. Head matches, random targets and sink exclusions were fixed on the selection set and evaluated on ordinary and repeated passages, as in E8. The span basis removes the requirement for identical token boundaries, and all 45 pairs were scored on 95 evaluation passages.

Every pair retains a positive late gap over random heads, from +0.097 to +0.211, and higher matched similarity on the same text than on different texts, by +0.093 to +0.300. Twenty-four of the pairs cross the two tokenisers, each setting a Pythia model against one on the GPT-2 vocabulary, and their late gaps run +0.119 to +0.166, inside the range of the pairs E8 could score.

The depth profile reverses under span aggregation. Twenty-nine of the 45 pairs have a higher late than early gap, against two of 21 in E8, and early gaps run +0.079 to +0.200. Summing key weights within spans discards routing inside a span, where much of the early correspondence sits. The two experiments therefore measure different quantities and their depth profiles should not be read against each other.

The seed-only pair is highest at both depths, +0.200 early and +0.211 late, and has the largest same-minus-different difference at +0.300. Late gap correlates at +0.221 with log smaller-model size, −0.229 with log size ratio and +0.322 with mean excluded-head share, against +0.401, −0.347 and +0.212 over E8's 21 pairs and +0.700, −0.086 and +0.453 over the original eight-model panel's twelve. None of the three is stable to which pairs are scored. Per-model late means range from +0.132 to +0.159, over nine partners each.

Every late gap stays positive at sink cutoffs 0.7 and 1.0, although OPT against CRFM x49 falls to +0.006 at 0.7. Repeating each passage's first eight-token prefix lowers the late gap in 44 of the 45 pairs, by up to 0.076; Pythia-1b against GPT-Neo rises by 0.011.
