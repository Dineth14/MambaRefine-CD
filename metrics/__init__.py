"""Metrics package for MambaRefineCD.

Binary CD  → metrics.binary_cd_metrics.BinaryMetrics
SECOND SCD → metrics.second_scd_metrics.SECONDSCDMetrics
"""
from metrics.binary_cd_metrics import BinaryMetrics
from metrics.second_scd_metrics import SECONDSCDMetrics

__all__ = ["BinaryMetrics", "SECONDSCDMetrics"]
