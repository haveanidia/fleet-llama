# Fleet v0.12 public release checklist

- [ ] `modeling_fleet.py` loads with `trust_remote_code=True`.
- [ ] Fresh-process route sanity test passes.
- [ ] Router sidecar contains no NaN/Inf.
- [ ] Generation smoke test passes for code, math, medical, function-calling, SQL, and NULL/general.
- [ ] Model card explains turn-based semantic routing and the trained router + prior correction layer.
- [ ] Base-model and every expert license reviewed before redistributing weights.
- [ ] No training caches, tokens, logs, or private paths committed.
- [ ] Triton claims are benchmarked against the actual Fleet inference path, not merely import/runtime availability.
- [ ] Full-model-to-LoRA conversions are evaluated after SVD compression.
