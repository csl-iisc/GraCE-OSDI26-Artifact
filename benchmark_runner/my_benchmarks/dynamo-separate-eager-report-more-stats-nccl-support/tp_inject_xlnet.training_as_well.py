# tp_inject_xlnet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ---- tiny TP linears (Megatron-style) ----

def _rk_ws(group):
    return dist.get_rank(group), dist.get_world_size(group)

class ColumnParallelLinear(nn.Module):
    """
    Split OUT features across ranks; output stays sharded along last dim.
    Parameters are allocated on the provided device/dtype to avoid CPU/CUDA mismatch.
    """
    def __init__(self, in_features, out_features, bias=True, group=None, device=None, dtype=None):
        super().__init__()
        assert group is not None
        self.group = group
        rk, ws = _rk_ws(group)
        assert out_features % ws == 0, "out_features must be divisible by TP world size"
        self.out_per_rank = out_features // ws

        self.weight = nn.Parameter(
            torch.empty(self.out_per_rank, in_features, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.empty(self.out_per_rank, device=device, dtype=dtype)
        ) if bias else None
        self.reset_parameters(in_features)

    def reset_parameters(self, in_features):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Use F.linear instead of x.matmul(self.weight.t()) for Inductor friendliness
        return F.linear(x, self.weight, self.bias)  # sharded last dim

class RowParallelLinear(nn.Module):
    """
    Split IN features across ranks; local partial outputs are all-reduced (SUM)
    to replicate the full output on every rank.
    """
    def __init__(self, in_features, out_features, bias=True, group=None, device=None, dtype=None):
        super().__init__()
        assert group is not None
        self.group = group
        rk, ws = _rk_ws(group)
        assert in_features % ws == 0, "in_features must be divisible by TP world size"
        self.in_per_rank = in_features // ws

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_per_rank, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype)
        ) if bias else None
        self.reset_parameters(in_features)

    def reset_parameters(self, in_features):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_sharded):
        # local matmul
        y_partial = F.linear(x_sharded, self.weight, None)
        # fuse across tensor-parallel ranks
        dist.all_reduce(y_partial, op=dist.ReduceOp.SUM, group=self.group)
        if self.bias is not None:
            y_partial = y_partial + self.bias
        return y_partial  # replicated full output

# ---- TP versions of XLNet submodules ----

def _chunk(param, tp, dim):
    # returns tensor chunk (no grad history) for this rank
    return torch.chunk(param.data, tp, dim=dim)

class XLNetRelativeAttentionTP(nn.Module):
    """
    Head-parallel attention for HF XLNet: shards q/k/v/r/o on head dim, biases per-head, seg_embed on head dim.
    One all-reduce after out-projection to match original results.
    """
    def __init__(self, base_attn, tp_group):
        super().__init__()
        self.base = base_attn  # reuse dropout, layer_norm, scalars, config fields
        self.tp_group = tp_group
        self.tp_rank = dist.get_rank(tp_group)
        self.tp_world = dist.get_world_size(tp_group)

        assert self.base.n_head % self.tp_world == 0, "n_head must be divisible by TP size"

        # shard parameters (clone preserves device/dtype of base params)
        r = self.tp_rank; tp = self.tp_world
        self.q = nn.Parameter(_chunk(self.base.q, tp, dim=1)[r].contiguous().clone())
        self.k = nn.Parameter(_chunk(self.base.k, tp, dim=1)[r].contiguous().clone())
        self.v = nn.Parameter(_chunk(self.base.v, tp, dim=1)[r].contiguous().clone())
        self.o = nn.Parameter(_chunk(self.base.o, tp, dim=1)[r].contiguous().clone())
        self.r = nn.Parameter(_chunk(self.base.r, tp, dim=1)[r].contiguous().clone())
        self.r_r_bias = nn.Parameter(_chunk(self.base.r_r_bias, tp, dim=0)[r].contiguous().clone())
        self.r_s_bias = nn.Parameter(_chunk(self.base.r_s_bias, tp, dim=0)[r].contiguous().clone())
        self.r_w_bias = nn.Parameter(_chunk(self.base.r_w_bias, tp, dim=0)[r].contiguous().clone())
        self.seg_embed = nn.Parameter(_chunk(self.base.seg_embed, tp, dim=1)[r].contiguous().clone())

        # cache a few attrs (modules reused; each rank has its own copy)
        self.n_head = self.base.n_head
        self.d_head = self.base.d_head
        self.d_model = self.base.d_model
        self.scale = self.base.scale
        self.dropout = self.base.dropout
        self.layer_norm = self.base.layer_norm

    @staticmethod
    def _einsum_h(x, W):  # [i,b,h,d] or [i,b,d_model] with W [d_model,h_local,d]
        return torch.einsum("ibh,hnd->ibnd", x, W)

    def rel_shift_bnij(self, x, klen=-1):
        return self.base.rel_shift_bnij(x, klen=klen)

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r,
                      seg_mat=None, attn_mask=None, head_mask=None, output_attentions=False):
        if isinstance(head_mask, torch.Tensor):
            nh_local = self.n_head // self.tp_world
            start = self.tp_rank * nh_local
            head_mask = head_mask[..., start:start+nh_local].contiguous()

        ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum("ijbs,ibns->bnij", seg_mat, ef)

        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum("ijbn->bnij", attn_mask)
            else:
                attn_score = attn_score - 1e30 * torch.einsum("ijbn->bnij", attn_mask)

        attn_prob = torch.nn.functional.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * torch.einsum("ijbn->bnij", head_mask)

        attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        if output_attentions:
            return attn_vec, torch.einsum("bnij->ijbn", attn_prob)
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        # partial over local heads, then fuse across TP ranks
        attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        dist.all_reduce(attn_out, op=dist.ReduceOp.SUM, group=self.tp_group)
        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        return self.layer_norm(attn_out)

    def forward(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat,
                mems=None, target_mapping=None, head_mask=None, output_attentions=False):
        # ---- training-time grad all-reduce only for the column-parallel branch ----
        # We create a view of h/g that is used ONLY by the Q/K/V projections.
        # The residual path still uses the original h/g, so its grad won't be summed twice.
        if self.training and h is not None and h.requires_grad:
            h_tp = h.view_as(h)
            def _ar_h(gh):
                dist.all_reduce(gh, op=dist.ReduceOp.SUM, group=self.tp_group)
                return gh
            h_tp.register_hook(_ar_h)
        else:
            h_tp = h

        if self.training and g is not None and g.requires_grad:
            g_tp = g.view_as(g)
            def _ar_g(gg):
                dist.all_reduce(gg, op=dist.ReduceOp.SUM, group=self.tp_group)
                return gg
            g_tp.register_hook(_ar_g)
        else:
            g_tp = g
        # --------------------------------------------------------------------------
        if g is not None:
            # Only the h-part of cat needs TP grad reduction; use h_tp in cat.
            cat = torch.cat([mems, h_tp], dim=0) if (mems is not None and mems.dim() > 1) else h_tp

            k_head_h = self._einsum_h(cat, self.k)
            v_head_h = self._einsum_h(cat, self.v)
            k_head_r = self._einsum_h(r.type(self.r.dtype), self.r)

            q_head_h = self._einsum_h(h_tp, self.q)
            attn_vec_h = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r,
                                            seg_mat=seg_mat, attn_mask=attn_mask_h,
                                            head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h
            output_h = self.post_attention(h, attn_vec_h)

            q_head_g = self._einsum_h(g_tp, self.q)
            if target_mapping is not None:
                q_head_g = torch.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r,
                                                seg_mat=seg_mat, attn_mask=attn_mask_g,
                                                head_mask=head_mask, output_attentions=output_attentions)
                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
                attn_vec_g = torch.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r,
                                                seg_mat=seg_mat, attn_mask=attn_mask_g,
                                                head_mask=head_mask, output_attentions=output_attentions)
                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
            output_g = self.post_attention(g, attn_vec_g)
            outputs = (output_h, output_g)
            if output_attentions:
                outputs = outputs + ((attn_prob_h, attn_prob_g),)
            return outputs

        # single-stream (no g)
        cat = torch.cat([mems, h_tp], dim=0) if (mems is not None and mems.dim() > 1) else h_tp
        q_head_h = self._einsum_h(h_tp, self.q)
        k_head_h = self._einsum_h(cat, self.k)
        v_head_h = self._einsum_h(cat, self.v)
        k_head_r = self._einsum_h(r.type(self.r.dtype), self.r)
        attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r,
                                      seg_mat=seg_mat, attn_mask=attn_mask_h,
                                      head_mask=head_mask, output_attentions=output_attentions)
        if output_attentions:
            attn_vec, attn_prob = attn_vec
        output_h = self.post_attention(h, attn_vec)
        outputs = (output_h, None)
        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs

class XLNetFeedForwardTP(nn.Module):
    """
    Column-parallel fc1 (out split), Row-parallel fc2 (in split + all-reduce).
    Crucially, we allocate TP params on the same device/dtype as the base FFN.
    """
    def __init__(self, base_ff, tp_group):
        super().__init__()
        assert dist.is_initialized()
        self.tp_group = tp_group
        self.dropout = base_ff.dropout
        self.layer_norm = base_ff.layer_norm
        self.act = base_ff.activation_function

        # read sizes & device/dtype from base
        d_model = base_ff.layer_2.out_features
        d_inner = base_ff.layer_1.out_features
        dev = base_ff.layer_1.weight.device
        dtp = base_ff.layer_1.weight.dtype

        # build TP linears with correct device/dtype
        self.fc1 = ColumnParallelLinear(d_model, d_inner, bias=True, group=tp_group, device=dev, dtype=dtp)
        self.fc2 = RowParallelLinear(d_inner, d_model, bias=True, group=tp_group, device=dev, dtype=dtp)

        # copy shards from base_ff (clone keeps device/dtype)
        tp = dist.get_world_size(tp_group); rk = dist.get_rank(tp_group)

        # fc1: split OUT features across ranks (dim=0)
        w1 = base_ff.layer_1.weight.detach()  # [d_inner, d_model]
        b1 = base_ff.layer_1.bias.detach()    # [d_inner]
        w1_shard = torch.chunk(w1, tp, dim=0)[rk].contiguous()
        b1_shard = torch.chunk(b1, tp, dim=0)[rk].contiguous()
        with torch.no_grad():
            self.fc1.weight.copy_(w1_shard)
            self.fc1.bias.copy_(b1_shard)

        # fc2: split IN features across ranks (dim=1)
        w2 = base_ff.layer_2.weight.detach()  # [d_model, d_inner]
        b2 = base_ff.layer_2.bias.detach()    # [d_model]
        w2_shard = torch.chunk(w2, tp, dim=1)[rk].contiguous()
        with torch.no_grad():
            self.fc2.weight.copy_(w2_shard)
            self.fc2.bias.copy_(b2)  # bias applied after all-reduce => full copy per rank

    def forward(self, inp):
        # ---- training-time grad all-reduce only for the TP branch into fc1 ----
        if self.training and inp.requires_grad:
            x_tp = inp.view_as(inp)
            def _ar_inp(g):
                dist.all_reduce(g, op=dist.ReduceOp.SUM, group=self.tp_group)
                return g
            x_tp.register_hook(_ar_inp)
        else:
            x_tp = inp
        # -----------------------------------------------------------------------

        y = self.fc1(x_tp)       # sharded (column-parallel)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)          # replicated
        y = self.dropout(y)
        return self.layer_norm(y + inp)

# ---- public API: apply to an already-loaded HF XLNet ----

def build_tp_group(tp_size: int):
    ws = dist.get_world_size()
    assert ws % tp_size == 0, "world_size must be multiple of tp_size"
    if ws == tp_size:
        return dist.group.WORLD
    rk = dist.get_rank()
    dp_id = rk // tp_size
    start = dp_id * tp_size
    ranks = list(range(start, start + tp_size))
    return dist.new_group(ranks=ranks)

def apply_xlnet_tensor_parallel(model, tp_group):
    """
    In-place surgery: swap each XLNetLayer's rel_attn & ff with TP versions, sharding weights.
    Call AFTER loading pretrained weights, AFTER model.to(device), and BEFORE torch.compile.
    """
    from transformers.models.xlnet.modeling_xlnet import XLNetModel

    # descend to the core transformer (some heads wrap it as .transformer)
    core = model
    if not isinstance(core, XLNetModel):
        assert hasattr(model, "transformer"), "Expected a .transformer with XLNetModel"
        core = model.transformer

    ws = dist.get_world_size(tp_group)
    assert core.layer[0].rel_attn.n_head % ws == 0, "n_head must be divisible by TP size"
    d_inner = core.layer[0].ff.layer_1.out_features
    assert d_inner % ws == 0, "d_inner must be divisible by TP size"

    for layer in core.layer:
        layer.rel_attn = XLNetRelativeAttentionTP(layer.rel_attn, tp_group)
        layer.ff       = XLNetFeedForwardTP(layer.ff, tp_group)
    return model
