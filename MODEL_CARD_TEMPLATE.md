---
library_name: transformers
pipeline_tag: text-generation
base_model: meta-llama/Llama-3.2-3B-Instruct
---

# Fleet Llama 3.2 3B v0.12

Fleet v0.12 is a turn-routed specialist system built around a shared Llama 3.2 3B backbone.

## Routing

Fleet v0.12 uses **turn-level semantic routing**. A normalized representation of the user turn is passed through a trained router head. Small domain priors correct strongly signaled code, math, medical, function-calling, and SQL prompts. The final expert route is selected once and reused during generation.

This is not keyword-only routing and it is not per-token routing.

## Loading

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "YOUR_USERNAME/Fleet-Llama-3.2-3B-v0.12"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, device_map="auto")
```

## Experts

Document each included expert, its source repository, training data if known, and license here before publishing weights.

## Validation

Record the fresh-process 6-route sanity test, generation smoke test, router finite check, and any real throughput benchmark here.

## License

The model repository must preserve and comply with the licenses of the Llama base model and every redistributed adapter or derivative weight artifact. The Apache-2.0 license in the source-code repository applies only to Fleet source tooling, not automatically to model weights.
