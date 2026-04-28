# torchrun --nproc_per_node=2 check_xlnet_tp_equiv.py
import copy, torch, torch.distributed as dist
from transformers import XLNetLMHeadModel, XLNetConfig

def bcast(x):
    dist.broadcast(x, src=0)
    return x

def same(x, y, name, rtol=1e-5, atol=1e-6):
    x32, y32 = x.float(), y.float()
    ok = torch.allclose(x32, y32, rtol=rtol, atol=atol)
    err = (x32 - y32).abs()
    print(f"[{name}] allclose={ok} max_abs={err.max().item():.4e} mean_abs={err.mean().item():.4e}")
    return ok

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)

    # Build & load a tiny model (or from pretrained); important: eval + float32 for tight check
    cfg = XLNetConfig.from_pretrained("xlnet/xlnet-large-cased")
    dense = XLNetLMHeadModel.from_pretrained("xlnet/xlnet-large-cased").to(rank).eval().bfloat16()

    # Make a deepcopy BEFORE TP surgery for the dense reference
    ref = copy.deepcopy(dense)

    # Create same input on rank0 and broadcast
    bs, seqlen = 1, 16
    if rank == 0:
        inp = torch.randint(5, 5000, (bs, seqlen), device=f"cuda:{rank}")
        attn_mask = torch.ones_like(inp)
    else:
        inp = torch.empty(bs, seqlen, dtype=torch.long, device=f"cuda:{rank}")
        attn_mask = torch.empty_like(inp)
    bcast(inp)
    bcast(attn_mask)

    # Get dense (TP=1) reference on rank0 only
    with torch.no_grad():
        if rank == 0:
            ref_out = ref(input_ids=inp, attention_mask=attn_mask, output_hidden_states=True).logits
        else:
            ref_out = torch.empty(bs, seqlen, cfg.vocab_size, device=f"cuda:{rank}")
    # broadcast reference so every rank can compare locally
    dist.broadcast(ref_out, src=0)

    # --- apply TP injection in-place (AFTER weights loaded & .to(device)) ---
    from tp_inject_xlnet import apply_xlnet_tensor_parallel, build_tp_group
    tp_group = build_tp_group(tp_size=2)
    dense_tp = apply_xlnet_tensor_parallel(dense, tp_group)
    dense_tp.eval().float()

    # Forward TP model (identical inputs on all ranks)
    tp_out = dense_tp(input_ids=inp, attention_mask=attn_mask, output_hidden_states=True).logits
    print(f"{ref(input_ids=inp, attention_mask=attn_mask, output_hidden_states=True)=}")
    print(f"{dense_tp(input_ids=inp, attention_mask=attn_mask, output_hidden_states=True)=}")
    # Compare logits
    print(f"Rank {rank}: comparing logits to dense reference...")
    same(tp_out, ref_out, "logits (bf16 users can relax to rtol=1e-2, atol=3e-2)")

    # Also compare per-layer hidden states to localize issues
    ref_hs = ref(input_ids=inp, attention_mask=attn_mask, output_hidden_states=True).hidden_states
    tp_hs  = dense_tp(input_ids=inp, attention_mask=attn_mask, output_hidden_states=True).hidden_states
    for i, (a, b) in enumerate(zip(tp_hs, ref_hs)):
        same(a, b, f"hidden_state[{i}]")

    # Cross-rank consistency (rank0 vs rank1)
    # gather rank1 logits to rank0 and compare
    if rank == 0:
        other = torch.empty_like(tp_out)
        dist.recv(other, src=1)
        same(tp_out, other, "logits rank0==rank1")
    else:
        dist.send(tp_out, dst=0)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    
