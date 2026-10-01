# Fleet v0.12

Fleet v0.12 is a **turn-routed expert system** built around a shared Llama 3.2 3B backbone and specialist adapters.

## What changed in v0.12

Fleet v0.12 routes **once per user turn**, not once per generated token. The route is reused for the response, reducing routing overhead on longer generations.

Routing combines:

1. a **semantic turn representation** from the model's token embeddings,
2. a **trained router head** stored in `fleet_router.safetensors`, and
3. small **domain priors** for obvious code, math, medical, tool-calling, and SQL prompts.

So yes: **Fleet v0.12 uses semantics**. The trained router remains the main learned signal; the lexical priors are only a correction layer.

```text
prompt
  -> token embeddings
  -> layer-0 normalization
  -> pooled normalized turn vector
  -> trained router head
  -> small domain-prior adjustment
  -> one route for the whole turn
  -> selected specialist / NULL
```

## Reference expert layout

The v0.12 release used a shared `meta-llama/Llama-3.2-3B-Instruct` backbone with specialist adapters such as code, medical, math, function-calling, and SQL.

This repository does **not** redistribute upstream model weights. Model weights and adapters remain subject to their original licenses.

## Install

```bash
python -m pip install -e .
```

Optional Hugging Face support:

```bash
python -m pip install -e '.[hub]'
```

## CLI

```bash
fleet expert list
fleet expert install --name sql --source BY-ALF/llama-3.2-3b-sql-lora
fleet expert uninstall --name sql
```

## Convert a fully fine-tuned model into a LoRA-style Fleet expert

If a specialist is a **full fine-tune of the same exact base architecture**, Fleet can compress the full-model delta into low-rank matrices.

```bash
fleet convert-full-to-lora \
  --base meta-llama/Llama-3.2-3B-Instruct \
  --specialist your-org/your-full-finetune \
  --output ./experts/my-specialist \
  --rank 32
```

Conceptually:

```text
Delta = W_specialist - W_base
Delta ~= B @ A
```

The converter uses truncated SVD per selected 2-D weight matrix. Higher rank preserves more of the full fine-tune but produces a larger adapter.

### Important converter constraints

The base and specialist must have matching tensor names and shapes. A Llama 3.2 3B expert can be converted against the same Llama 3.2 3B base. You cannot directly subtract and convert a Qwen, Mistral, different-width Llama, or different-depth model into this backbone.

## Python routing API

For a Fleet-compatible model exposing `fleet_route_probs`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "YOUR_USERNAME/Fleet-Llama-3.2-3B-v0.12"

tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    trust_remote_code=True,
    device_map="auto",
)

prompt = "Solve 7x - 5 = 30 and show the steps."
text = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
enc = tok(text, return_tensors="pt")
dev = model.get_input_embeddings().weight.device
enc = {k: v.to(dev) for k, v in enc.items()}

probs = model.fleet_route_probs(
    enc["input_ids"],
    enc.get("attention_mask"),
    prompt_text=prompt,
)[0]
```

## Release validation

```bash
fleet validate-release --model-dir /path/to/Fleet-Llama-3.2-3B-v0.12
```

The validator checks:

- router sidecar exists,
- all router tensors are finite,
- configuration metadata is readable,
- optional Triton availability,
- expected Fleet runtime files are present.

## Triton

Triton being installed does **not** automatically make Fleet faster. It only improves inference if Fleet's hot path actually calls Triton kernels. The included Triton check verifies that a real GPU kernel can compile and execute; benchmark the actual model path separately before making a speed claim.

## Repository structure

```text
src/fleet_v012/
  cli.py
  converter.py
  experts.py
  routing.py
  validate.py
examples/
  route_once.py
  convert_full_model.py
scripts/
  triton_check.py
```

## License

The source tooling in this repository is released under Apache-2.0. **Model weights are not covered by this source license.** Respect Meta Llama and every expert/checkpoint's upstream license and attribution requirements.
