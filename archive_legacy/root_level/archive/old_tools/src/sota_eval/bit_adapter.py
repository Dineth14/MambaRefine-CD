from __future__ import annotations

from .base_adapter import AdapterUnavailableError, BaseAdapter


class BITAdapter(BaseAdapter):
    model_name = "BIT"
    expected_normalization = "imagenet"
    expected_output_type = "two_class_logits"

    def build_model(self):
        module = self.import_external_models_module("models.networks")
        if not hasattr(module, "define_G"):
            raise AdapterUnavailableError("BIT models.networks.define_G is missing")
        args = self.namespace(
            net_G="base_transformer_pos_s4_dd8_dedim8",
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
            raise AdapterUnavailableError(f"BIT forward failed: {exc}") from exc
