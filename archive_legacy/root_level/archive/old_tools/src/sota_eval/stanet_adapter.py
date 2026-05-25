from __future__ import annotations

import inspect

from .base_adapter import AdapterUnavailableError, BaseAdapter


class STANetAdapter(BaseAdapter):
    model_name = "STANet"
    expected_normalization = "imagenet"
    expected_output_type = "two_class_logits"

    def build_model(self):
        module = self.import_module_candidates(
            ["models.networks", "models", "network"],
            ["models/networks.py", "models.py", "network.py"],
        )
        for name in ["CDNet34", "CDNet", "STANet"]:
            cls = getattr(module, name, None)
            if inspect.isclass(cls):
                for kwargs in ({"output_nc": 2}, {"num_classes": 2}, {}):
                    try:
                        return cls(**kwargs).to(self.device)
                    except Exception:
                        continue
        raise AdapterUnavailableError("STANet model class was not identified automatically")

    def load_checkpoint(self, model):
        checkpoint = self.load_torch_checkpoint()
        state_dict = self.strip_prefixes(self.extract_state_dict(checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}

    def preprocess_batch(self, batch):
        batch = dict(batch)
        batch["image_a"] = batch["image_a"].to(self.device)
        batch["image_b"] = batch["image_b"].to(self.device)
        return batch

    def forward(self, model, batch):
        try:
            return model(batch["image_a"], batch["image_b"])
        except Exception as exc:
            raise AdapterUnavailableError(f"STANet forward failed: {exc}") from exc
