from __future__ import annotations

from .base_adapter import AdapterUnavailableError, BaseAdapter


class ChangeFormerAdapter(BaseAdapter):
    model_name = "ChangeFormer"
    expected_normalization = "half"
    expected_output_type = "two_class_logits"

    def build_model(self):
        module = self.import_external_models_module("models.networks")
        if not hasattr(module, "define_G"):
            raise AdapterUnavailableError("ChangeFormer models.networks.define_G is missing")
        args = self.namespace(
            net_G="ChangeFormerV6",
            embed_dim=256,
            gpu_ids=[],
            n_class=2,
            img_size=256,
            checkpoint_dir="",
            vis_dir="",
            batch_size=1,
            split="test",
            data_name="LEVIR",
        )
        model = module.define_G(args, gpu_ids=[])
        return model.to(self.device)

    def load_checkpoint(self, model):
        checkpoint = self.load_torch_checkpoint()
        state_dict = self.strip_prefixes(self.extract_state_dict(checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        return {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

    def preprocess_batch(self, batch):
        batch = dict(batch)
        batch["image_a"] = self.maybe_re_normalize(batch["image_a"].to(self.device), "half")
        batch["image_b"] = self.maybe_re_normalize(batch["image_b"].to(self.device), "half")
        return batch

    def forward(self, model, batch):
        try:
            return model(batch["image_a"], batch["image_b"])
        except Exception as exc:
            raise AdapterUnavailableError(f"ChangeFormer forward failed: {exc}") from exc
