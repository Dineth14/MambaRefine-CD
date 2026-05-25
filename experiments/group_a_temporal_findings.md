# Group A Temporal Findings

## Goal

Evaluate which temporal difference representation entering D-RBI is most effective while keeping the encoder, D-RBI, ARF-FPN decoder, boundary refinement, losses, metrics, and data protocol fixed.

## Tested Variants

### A0_abs_only

`D_in^s = |F2^s - F1^s|`

Tests whether sign-invariant temporal magnitude is sufficient.

### A1_signed_only

`D_in^s = F2^s - F1^s`

Tests whether temporal direction alone is sufficient.

### A2_abs_signed

`D_in^s = [|F2^s - F1^s|, F2^s - F1^s]`

Tests the default difference-only temporal design.

## Hypotheses

H1: Absolute difference should be stable for binary change detection because both appearance and disappearance map to the changed class.

H2: Signed difference may help when temporal direction matters, but may be weaker for direction-invariant binary change detection.

H3: Absolute plus signed difference should preserve both magnitude and direction without exposing raw unchanged semantic features to D-RBI.

## Results

Fill after experiments.

## Final Decision

Keep / remove / modify temporal input design.
