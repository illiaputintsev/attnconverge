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

GPT-2, GPT-2-medium, GPT-Neo-125m and OPT-125m were compared on the same fifty sentences at a 0.9 sink cutoff. The four models give six pairs: three approximately equal in size and three 2.8–2.9x apart. For OPT, the leading BOS row and column were removed and the remaining rows renormalised, so its scores describe attention conditional on the sentence tokens.

All six pairs have a lower late than early gap, but every late average remains positive, between +0.106 and +0.155. The three approximately equal-sized pairs decline by +0.115 to +0.171. GPT-2 against GPT-2-medium declines by +0.142, from +0.289 to +0.147, within that range despite its 2.9x size ratio. A decline with depth therefore occurs without size mismatch, while unequal sizes can still retain a positive late gap.

The equal-sized pairs have a mean decline of +0.134 and a late gap of +0.150, against +0.162 and +0.122 for the unequal pairs. The larger decline in the unequal group leaves size mismatch as a possible contributor. All three unequal pairs involve GPT-2-medium, however, so the comparison also depends on that particular model. Early and late mean the first two and last three source layers; these cover narrower depth intervals when the 24-layer GPT-2-medium is the source. The group difference cannot be attributed to parameter count alone.

Early gaps across organisations range from +0.253 to +0.321, averaging +0.283, compared with +0.289 for the two GPT-2 sizes. Early correspondence is therefore present across each pairing of OpenAI, EleutherAI and Meta models.

GPT-2 against GPT-Neo has an early gap of +0.321 here and +0.302 in E6. The best-match scores agree at the reported precision; the difference comes from the random-pair baseline being sampled again.

These gaps describe best matches selected and evaluated on the same sentences. The random-pair baseline does not account for choosing the best of several candidates, so a positive gap alone does not establish that the correspondence generalises to new text or depends on the input.
