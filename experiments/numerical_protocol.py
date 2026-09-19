"""Predeclared FP32 comparison policy (PyTorch assert_close defaults)."""
FP32_TOLERANCES = {'rtol': 1.3e-6, 'atol': 1e-5}
POLICY_SOURCE = 'https://docs.pytorch.org/docs/main/testing.html#torch.testing.assert_close'
PROTOCOL_VERSION = 'fp32-v6-seeds-0-through-9'

TIMING_PROTOCOL = {"seeds": list(range(10)), "warmup_seeds": [0, 1], "warmups": 2, "runs": 10}
