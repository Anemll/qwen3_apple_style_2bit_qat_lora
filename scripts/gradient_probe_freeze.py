# scripts/gradient_probe_freeze.py
import argparse, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qat_lora.quantizer import QATQuantConfig
from qat_lora.model_utils import replace_linear_with_qat
from qat_lora.topk_cache_dataset import TopKCacheDataset, topk_cache_collate

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    p.add_argument("--qat_checkpoint", required=True)
    p.add_argument("--kd_cache_dir", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32"])
    p.add_argument("--skip_lm_head", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--T", type=float, default=2.0)
    return p.parse_args()

def main():
    args = parse()
    device = torch.device(args.device)
    dtype = {"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32}[args.dtype]

    # Build student model (QATLinear replacements)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
    model.config.use_cache = False
    exclude = r"(^lm_head$)" if args.skip_lm_head else None
    replace_linear_with_qat(model, qc=QATQuantConfig(), exclude_regex=exclude, verbose=False)

    sd = torch.load(args.qat_checkpoint, map_location="cpu")
    # Accept either model-only dict or full checkpoint dict
    if isinstance(sd, dict) and "model" in sd and "optimizer" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device=device, dtype=dtype).train()

    # KD cache loader
    ds = TopKCacheDataset(args.kd_cache_dir, shuffle_files=True, seed=0)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, collate_fn=topk_cache_collate)

    # Simple KD-on-candidates loss using cached topk only
    W = model.lm_head.weight  # lm_head is FP if skip_lm_head
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)

    grad_accum = {}

    def add_grad(name, g):
        if g is None: return
        v = float(g.detach().float().norm().cpu())
        grad_accum[name] = grad_accum.get(name, 0.0) + v

    it = iter(dl)
    for step in range(args.steps):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        topk_idx = batch["topk_idx"].to(device)         # [B,S,K]
        topk_logits = batch["topk_logits"].to(device)   # [B,S,K]

        out = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state[:, :-1, :]       # [B,S,H]
        B,S,H = hidden.shape
        K = topk_idx.shape[-1]
        N = B*S

        h = hidden.reshape(N, H)
        idx = topk_idx.reshape(N, K)
        Wk = W[idx]                                     # [N,K,H]
        s_logits = torch.bmm(Wk, h.unsqueeze(-1)).squeeze(-1).reshape(B,S,K)

        T = args.T
        t = (topk_logits.float() / T)
        s = (s_logits.float() / T)
        p_t = torch.softmax(t, dim=-1)
        log_p_t = torch.log_softmax(t, dim=-1)
        log_p_s = torch.log_softmax(s, dim=-1)
        kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)    # [B,S]
        loss = (kl.mean() * (T*T))

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # collect grads
        for name, p in model.named_parameters():
            if p.grad is None: 
                continue
            if any(x in name for x in ["q_proj.weight","k_proj.weight","v_proj.weight","o_proj.weight",
                                       "gate_proj.weight","up_proj.weight","down_proj.weight","_f_param"]):
                add_grad(name, p.grad)

        opt.step()
        print(f"[step {step}] loss={float(loss.detach().cpu()):.4f}")

    # Report
    ranked = sorted(grad_accum.items(), key=lambda kv: kv[1], reverse=True)
    print("\n=== Top gradient norms (accumulated) ===")
    for n,v in ranked[:40]:
        print(f"{v:12.4f}  {n}")

if __name__ == "__main__":
    main()
