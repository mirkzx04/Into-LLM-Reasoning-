from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

def get_rope_theta(self):
    rp = getattr(self, "rope_parameters", None)

    if isinstance(rp, dict):
        return rp.get("rope_theta", 1_000_000.0)

    if rp is not None and hasattr(rp, "rope_theta"):
        return rp.rope_theta

    return 1_000_000.0

if not hasattr(Qwen2Config, "rope_theta"):
    Qwen2Config.rope_theta = property(get_rope_theta)


