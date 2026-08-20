# Observations

## E1. One sentence, both models

A single 12-token sentence was run through 70m and 160m. Both returned matrices of the same shape over the same tokens, with every row summing to one and the upper triangle at zero.

Row entropy declines with depth in both models, beginning around 1.1 nats and reaching 0.11 for 70m and 0.02 for 160m. Exponentiated, this means each token draws on roughly three tokens' worth of attention in the early layers and approximately one by the end. A head distributing its weight evenly across twelve tokens would place about 0.26 on token 0, so the late layers exceed that by a factor of three.

The two measurements describe the same phenomenon from different angles. Since every row must sum to one, weight accumulating on token 0 leaves less available elsewhere. Entropy falling with depth could be read as the heads narrowing onto whichever token carries the relevant information, but the per-head weights show the concentration landing on token 0 in every case.

At the level of individual heads, both first layers contain one positioned almost entirely on the diagonal (70m H3, 160m H7), alongside heads placing their weight one position back and others distributing it across recent tokens with distance decay. The first layer works on raw embeddings with no context attached yet, so position is most of what a head can use. Both final layers are dominated by sinks, though each retains a single stepped-diagonal head (70m H7, 160m H11).

## E2. Fifty sentences

The measurement was repeated across fifty sentences of comparable length, with entropy normalised so that rows of different lengths could be averaged together, and with baselines added to both plots.

The E1 shapes persisted across the set. Normalised entropy falls from 0.65 in 70m and 0.72 in 160m at the first layer to 0.09 and 0.02 at the last, while sink fraction rises from 0.22, the value an evenly distributed head would produce at these sentence lengths, to approximately 0.8. The second layer of 70m places only 0.18 on token 0, below the 0.22 an evenly distributed head produces, so the early layers are directing attention away from the first token rather than towards it.

Sink fraction varies substantially between heads inside a single layer. This is why the sink curve carries wide error bands from mid-depth onwards. In 70m layer 4, heads 2, 3 and 7 approach complete sinking while heads 0 and 5 remain near 0.5. In layer 5, heads 2 and 7 place almost no weight on token 0 while heads 3 and 6 place nearly all of it. In neither layer does the mean correspond to the behaviour of any individual head.

Both models exhibit the same structure: low sink fraction through the early layers, a sharp transition at approximately one third of the depth (70m layer 3 of 6, 160m layer 4 of 12), and a high band extending to the end. Heads cluster near zero or near one rather than distributing evenly between them.

## E3. Cross-model head matching, 70m and 160m

Every head in 70m was compared against every head in 160m by cosine similarity over their attention matrices, averaged across the fifty sentences, keeping the best match for each head in 70m. Layers were paired by relative depth. Head numbering follows from initialisation and means nothing across models, so the comparison runs all against all rather than by index. Heads above 0.7 sink fraction were excluded.

The random-pair baseline is roughly 0.50 at every depth, between 0.499 and 0.526. Results are therefore given as the gap over that baseline rather than as raw similarity.

Best match reaches 0.824 at the first layer pair and 0.809 at the second, gaps of +0.325 and +0.294 over the baseline. The gap falls to +0.125 and +0.178 at the two middle pairs, then to −0.272 at depth 0.8 and −0.050 at the last layer.

At depth 0.8 the comparison involves two heads against three, and at depth 1.0 five against three, so the collapse cannot be attributed to the models rather than the filter on this evidence.

Keeping column 0 raised every figure. At depth 0.6 the masked comparison gave 0.690 and the unmasked 0.860. Both models placed most of their late weight on the first token, so the unmasked figures included agreement that reflects the sink rather than anything either model learned.
