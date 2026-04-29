from __future__ import annotations

from .bit_adapter import BITAdapter
from .changeformer_adapter import ChangeFormerAdapter
from .mcd_adapter import MCDAdapter
from .snunet_adapter import SNUNetAdapter
from .stanet_adapter import STANetAdapter

ADAPTERS = {
    "changeformer": ChangeFormerAdapter,
    "bit": BITAdapter,
    "snunet": SNUNetAdapter,
    "stanet": STANetAdapter,
    "mcd": MCDAdapter,
}


def list_registered_adapters() -> list[str]:
    return sorted(ADAPTERS.keys())


def build_adapter(adapter_name: str, cfg: dict, model_cfg: dict, dataset_name: str, checkpoint_path: str | None, device):
    key = str(adapter_name).lower().strip()
    if key not in ADAPTERS:
        raise KeyError(f"unknown adapter: {adapter_name}")
    return ADAPTERS[key](cfg, model_cfg, dataset_name, checkpoint_path, device)
