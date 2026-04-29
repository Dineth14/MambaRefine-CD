# MambaRefine-CD Method

## 1. Overview

MambaRefine-CD is a binary remote-sensing change detection model for paired images. It uses a shared visual backbone, explicit temporal difference modeling, region-boundary interaction, an adaptive decoder, and residual boundary refinement to predict a single binary change map.

## 2. MambaVision Backbone

The pre-change and post-change images are encoded by a shared MambaVision backbone:

`F1 = E(I1)`

`F2 = E(I2)`

The shared encoder keeps both timestamps in the same feature space and provides multi-scale features to the change decoder.

## 3. Differential Region-Boundary Interaction

D-RBI fuses temporal features with region and boundary gates. The interaction tensor is:

`X = [F1, F2, |F2 - F1|, F2 - F1, F1 * F2]`

The region and boundary responses are:

`R = Gr * D`

`B = Gb * D`

where `D` is the temporal difference feature and `Gr`, `Gb` are learned gates.

## 4. Signed Temporal Difference

Signed temporal difference keeps directionality:

`Ds = F2 - F1`

This complements absolute difference, which captures magnitude but discards temporal direction.

## 5. Adaptive Receptive Field Decoder

The decoder aggregates multi-scale region and boundary features with multiple dilation rates. This improves sensitivity to small local changes and larger spatial structures without changing the input resolution.

## 6. Boundary Residual Refinement

Boundary residual refinement corrects coarse change logits near object edges:

`Pf = Pc + delta * tanh(Delta)`

where `Pc` is the coarse prediction and `Delta` is the learned residual.

## 7. Binary Change Detection Output

The active repository predicts one binary change logit map. The final probability is:

`P = sigmoid(Pf)`

Pixels with `P >= threshold` are predicted as changed.

## 8. Loss Function

The active binary loss combines BCE and Dice loss, with optional coarse and boundary terms when enabled by config:

`L = Lbce + Ldice + wc * Lcoarse + wb * Lboundary`

DSIFN-CD and WHU-CD experiments use the same binary loss family for fair comparison.

## 9. Metrics

Active binary change detection datasets report only:

- Pre
- Rec
- F1
- IoU
- OA
