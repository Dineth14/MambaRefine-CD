# MambaRefine-CD Architecture Explanation for the 8 Diagrams

This document explains the MambaRefine-CD architecture in a form that is useful for designing the eight website diagrams. It focuses only on the architecture, the motivation behind each block, and the role of each figure in the overall story.

## High-level idea

MambaRefine-CD is a bi-temporal remote sensing change detection model. It compares two aligned images of the same location captured at different times:

- Image A: earlier observation
- Image B: later observation

The goal is to predict which pixels correspond to real scene change.

This is harder than simple image subtraction because differences between two dates can come from several sources:

- true semantic change in the scene
- illumination variation
- seasonal appearance shift
- misalignment noise
- local texture variation
- uncertainty around object boundaries

MambaRefine-CD is built to separate these effects instead of compressing them into one raw difference map.

The architecture has four main stages:

1. A shared MambaVision encoder extracts multi-scale features from both images.
2. A Differential Region-Boundary Interaction module builds structured temporal difference features at each scale.
3. An Adaptive Receptive Field decoder fuses region-oriented evidence across scales.
4. A boundary residual head sharpens the final prediction around object edges.

The final output is a refined binary change probability map.

For the current implementation, two details matter if the diagrams are meant to be code-faithful rather than only conceptual:

- the feature product term is optional and is disabled in the active config
- the boundary refinement head currently uses the finest boundary feature map rather than an explicit multi-scale boundary aggregation block

## Why this architecture exists

Many change detection pipelines rely on a simple temporal interaction rule:

```text
D_naive = |F2 - F1|
```

That operation is easy to compute but too weak as the main representation. It mixes together:

- meaningful change information
- nuisance appearance variation
- interior region evidence
- boundary transitions

This creates a representation that is often noisy, spatially ambiguous, and not specialized for either broad regions or sharp edges.

The central idea of MambaRefine-CD is that region evidence and boundary evidence should not be forced through the same path.

## Full architecture overview

At a system level, the model can be summarized as:

```text
Image A, Image B
    -> shared MambaVision encoder
    -> paired multi-scale features at four levels
    -> D-RBI at each level
    -> region stream + boundary stream
    -> region stream into Adaptive RF decoder
    -> coarse prediction P_c
    -> finest boundary stream into residual refinement head
    -> refined prediction P_f
    -> sigmoid output Y
```

Each stage solves a different problem:

- The encoder builds strong feature representations.
- D-RBI computes structured temporal interaction.
- The decoder handles scale diversity.
- The boundary head improves contour quality.

## Stage 1: Shared MambaVision encoder

Both input images pass through the same hierarchical MambaVision backbone with shared weights.

This matters because:

- both time steps are embedded into the same feature space
- temporal comparison becomes more meaningful
- parameters remain efficient because the encoder is not duplicated

The encoder outputs a pyramid of multi-scale features:

```text
F1_1, F1_2, F1_3, F1_4
F2_1, F2_2, F2_3, F2_4
```

where `F1_l` and `F2_l` are the feature maps for the two time steps at scale `l`.

These levels typically correspond to progressively coarser strides such as 4, 8, 16, and 32.

The shallow levels preserve high spatial detail, which is useful for boundaries and small structures. The deep levels preserve stronger semantic context, which is useful for understanding large changed regions. Both are necessary for robust change detection.

## Stage 2: Differential Region-Boundary Interaction (D-RBI)

At each scale, the model receives paired encoder features `F1_l` and `F2_l`.

Instead of relying only on the absolute difference, it constructs a richer temporal representation:

```text
Z_l = concat(F1_l, F2_l, |F2_l - F1_l|, F1_l * F2_l)
D_l = phi_l(Z_l)
```

Conceptually, that four-term form is the full intended D-RBI formulation. In the current training configuration, however, the product term is turned off for stability, so the active implementation is closer to:

```text
Z_l = concat(F1_l, F2_l, |F2_l - F1_l|)
D_l = phi_l(Z_l)
```

This representation is important because each term contributes different information:

- `F1_l` preserves pre-change context
- `F2_l` preserves post-change context
- `|F2_l - F1_l|` captures direct discrepancy
- `F1_l * F2_l` preserves correlation and agreement structure

The concatenated tensor `Z_l` is then compressed and mixed by a learned transformation `phi_l`, producing `D_l`, a structured temporal difference feature.

This is the core improvement over naive differencing. The model is no longer limited to a single raw subtraction signal.

So if a diagram is meant to show the conceptual method, including the product term is acceptable. If it is meant to match the currently active code path exactly, the product branch should be marked as optional or disabled by default.

## Stage 3: Region and boundary decomposition

After forming `D_l`, the model separates it into two specialized streams.

### Region stream

The region stream is produced by a gate that reads `D_l` directly:

```text
G_r = bounded sigmoid gate on D_l
R_l = G_r * D_l
```

Its purpose is to emphasize coherent interior change evidence. The gate is bounded so it remains stable and does not fully suppress features too aggressively.

The region stream `R_l` carries broad, spatially consistent information about changed areas. This is the main source of information used by the decoder.

### Boundary stream

The boundary stream is produced by a separate gate that depends on an edge-sensitive transformation such as Sobel magnitude:

```text
G_b = bounded sigmoid gate on Sobel(D_l)
B_l = G_b * D_l
```

Its purpose is different from the region stream. It highlights fine spatial transitions, contour cues, and edge-localized evidence.

The boundary gate is usually bounded more conservatively so that it acts as a focused refinement signal rather than a dominant feature path.

### Why the split matters

This split is one of the defining ideas in MambaRefine-CD.

Interior regions and object boundaries have different statistical behavior:

- region evidence is broad and spatially smooth
- boundary evidence is sparse and high-frequency

Trying to force both through one undifferentiated difference map makes learning harder. Separating them allows the network to specialize each path for a more precise role.

## Stage 4: Adaptive Receptive Field decoder

All region streams `{R_l}` are passed to the decoder.

The decoder is not a fixed plain FPN. It adapts its receptive field so that different spatial locations can rely on different context sizes:

```text
P_c = ARF_Decoder({R_l})
```

This is commonly represented with multiple dilation branches, such as:

```text
d = 1, 2, 4, 8
```

These branches let the decoder capture different spatial extents:

- small dilation for fine local details
- medium dilation for moderate structures
- large dilation for broad context

This matters because remote sensing changes vary widely in size. A single fixed receptive field is usually too rigid.

The decoder aggregates the region-oriented evidence and produces a coarse prediction `P_c`. This coarse map is usually strong in identifying changed regions but may still be imperfect at boundaries.

## Stage 5: Boundary residual refinement

The boundary stream is used after the coarse map is formed. Instead of producing a new prediction from scratch, it predicts a controlled correction:

```text
P_f = P_c + alpha * tanh(BoundaryHead({B_l}, P_c, Grad(P_c)))
Y   = sigmoid(P_f)
```

Conceptually, this notation describes a boundary-aware refinement stage informed by boundary features and coarse prediction structure. In the current implementation, the refinement head uses the finest boundary feature map, the coarse logits, and a Sobel edge map computed from the coarse probability:

```text
delta = BoundaryHead(B_1, P_c, Sobel(sigmoid(P_c)))
P_f   = P_c + 0.05 * tanh(delta)
Y     = sigmoid(P_f)
```

This design is deliberate.

The coarse prediction `P_c` already captures most of the semantic region decision. The boundary head only needs to refine the contour. Predicting a residual rather than a full mask has several advantages:

- it preserves stable interior predictions
- it reduces the risk of over-correction
- it makes the boundary branch a specialist module instead of a competing decoder

The correction is bounded through `tanh` and a small residual scale. In the active config, that scale is `0.05`. This keeps the boundary refinement focused and stable.

So the final output should be interpreted as:

- a coarse region-aware prediction
- plus a small edge-aware correction

## Stability and optimization logic

Some parts of the architecture are motivated by training stability as much as by representation quality.

Examples include:

- normalization before feature concatenation
- bounded gate outputs
- clamped gate logits
- bounded residual refinement
- conservative residual scaling, currently `0.05`

These constraints matter because the model combines multi-scale features, temporal interactions, edge-sensitive operators, and residual corrections. Without explicit control, this combination can become unstable during optimization.

## How the 8 diagrams map to the architecture

The eight figures are eight views of the same architectural story. Each one explains a different layer of reasoning.

## Diagram 01: Why naive differencing fails

This diagram is the motivation figure.

It should explain that simple temporal difference is not enough because it merges real change, nuisance variation, and edge uncertainty into one undifferentiated signal.

The reader should understand from this figure why a more structured temporal interaction mechanism is needed.

## Diagram 02: Shared MambaVision encoder

This diagram isolates the backbone.

It should explain that both images are processed by the same encoder and that the encoder produces aligned multi-scale features. The message is that temporal comparison happens in a learned hierarchical feature space, not only at the pixel level.

## Diagram 03: D-RBI module detail

This diagram explains how one pair of multi-temporal features becomes a structured difference representation.

It should show that D-RBI uses the two original feature maps and their absolute difference, and can optionally include their element-wise product. These are merged into `D_l`, which is a richer temporal representation than raw subtraction.

This figure should make clear that D-RBI is the main feature interaction block of the model.

## Diagram 04: Region and boundary gates

This diagram explains specialization.

Its purpose is to show that `D_l` is decomposed into two streams with different roles:

- region stream for coherent interior change evidence
- boundary stream for contour-sensitive evidence

This is one of the most important conceptual distinctions in the architecture.

## Diagram 05: Adaptive RF decoder

This diagram explains scale-adaptive region decoding.

It should show that region features from multiple scales are fused and that the decoder can use different receptive field sizes instead of a single fixed context window. The viewer should understand that this helps the model handle both small and large change patterns.

If the figure is meant to match the current code exactly, the boundary refinement subpart should show refinement driven by the finest boundary feature rather than an explicit multi-scale boundary fusion block.

## Diagram 06: Full end-to-end architecture

This is the global pipeline figure.

It should connect the entire model from the two inputs to the final refined output. This is the diagram that answers the main question: how does the complete architecture operate from beginning to end?

For implementation-level accuracy, this figure should also reflect that the current decoder uses all region scales for coarse prediction but only the finest boundary scale for residual refinement.

## Diagram 07: Design evolution timeline

This diagram explains research logic rather than dataflow.

It should show that the final model emerged step by step:

1. start with shared encoder and decoder baseline
2. improve temporal interaction
3. separate region and boundary evidence
4. add adaptive decoding and residual refinement

Its role is to show that the architecture is motivated and cumulative rather than arbitrary.

## Diagram 08: Metric explanation

This diagram explains how the model should be judged.

Because the no-change class usually dominates, overall accuracy and averaged metrics can look overly optimistic. The figure should explain why change-class metrics and boundary-sensitive metrics are necessary for a realistic evaluation of the architecture.

## What the viewer should understand after seeing all 8 diagrams

After seeing the full diagram set, the reader should understand the following architectural story:

1. Naive temporal differencing is too weak for real remote sensing change detection.
2. Shared multi-scale feature extraction creates a stable comparison space.
3. D-RBI builds richer temporal evidence than raw subtraction.
4. Region and boundary information are modeled separately because they behave differently.
5. Adaptive receptive field decoding improves scale robustness.
6. Boundary residual refinement improves contour quality without destabilizing region predictions.
7. The final network is a coherent sequence of motivated design decisions.
8. Evaluation should emphasize actual change quality and boundary quality, not only global scores.