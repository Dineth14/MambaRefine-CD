"""Decoder registry."""
from models.decoders.baseline_decoder    import BaselineDecoder
from models.decoders.adaptive_rf_decoder import AdaptiveRFDecoder
from models.decoders.refinement_decoder  import RefinementDecoder
from models.decoders.global_local_decoder import GlobalLocalDecoder

DECODER_REGISTRY = {
    "baseline":    BaselineDecoder,
    "adaptive_rf": AdaptiveRFDecoder,
    "refinement":  RefinementDecoder,
    "global_local": GlobalLocalDecoder,
}

__all__ = [
    "BaselineDecoder", "AdaptiveRFDecoder",
    "RefinementDecoder", "GlobalLocalDecoder",
    "DECODER_REGISTRY",
]
