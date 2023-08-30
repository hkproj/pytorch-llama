import torch
import torch.nn.functional as F
import math

def MultiheadAttentionBatched():
    d_model, seq_len_kv, seq_len, b, h, d_k, d_v = 512, 10, 10, 32, 8, (512 // 8), (512 // 8)

    X = torch.rand(b, seq_len, d_model)  # Query
    M = torch.rand(b, seq_len_kv, d_model)  # Key and Value
    mask = torch.rand(b, h, seq_len, seq_len_kv)
    P_q = torch.rand(h, d_model, d_k)  # W_q
    P_k = torch.rand(h, d_model, d_k)  # W_k
    P_v = torch.rand(h, d_model, d_v)  # W_v
    P_o = torch.rand(h, d_model, d_v)  # W_o
    
    Q = torch.einsum("bnd,hdk->bhnk ", X, P_q) 

    K = torch.einsum("bmd,hdk->bhmk", M, P_k)

    V = torch.einsum("bmd,hdv->bhmv", M, P_v)


    logits = torch.einsum("bhnk,bhmk->bhnm", Q, K)

    weights = torch.softmax(logits + mask, dim=-1)


    O = torch.einsum("bhnm,bhmv->bhnv ", weights, V)

    Y = torch.einsum("bhnv,hdv->bnd ", O, P_o)
    return Y


def MultiheadSelfAttentionIncremental():
    d_model, b, h, d_k, d_v = 512, 32, 8, (512 // 8), (512 // 8)

    m = 5  # Suppose we have already cached "m" tokens
    prev_K = torch.rand(b, h, m, d_k)
    prev_V = torch.rand(b, h, m, d_v)

    X = torch.rand(b, d_model)  # Query
    M = torch.rand(b, d_model)  # Key and Value
    P_q = torch.rand(h, d_model, d_k)  # W_q
    P_k = torch.rand(h, d_model, d_k)  # W_k
    P_v = torch.rand(h, d_model, d_v)  # W_v
    P_o = torch.rand(h, d_model, d_v)  # W_o

    q = torch.einsum("bd,hdk->bhk", X, P_q)
    new_K = torch.concat(
        [prev_K, torch.einsum("bd,hdk->bhk", M, P_k).unsqueeze(2)], axis=2
    )
    new_V = torch.concat(
        [prev_V, torch.einsum("bd,hdv->bhv", M, P_v).unsqueeze(2)], axis=2
    )
    logits = torch.einsum("bhk,bhmk->bhm", q, new_K)
    weights = torch.softmax(logits, dim=-1)
    O = torch.einsum("bhm,bhmv->bhv", weights, new_V)
    y = torch.einsum("bhv,hdv->bd", O, P_o)
    return y, new_K, new_V


def MultiquerySelfAttentionIncremental():
    d, b, h, k, v = 512, 32, 8, (512 // 8), (512 // 8)

    m = 5  # Suppose we have already cached "m" tokens
    prev_K = torch.rand(b, m, k)
    prev_V = torch.rand(b, m, v)

    X = torch.rand(b, d)  # Query
    M = torch.rand(b, d)  # Key and Value
    P_q = torch.rand(h, d, k)  # W_q
    P_k = torch.rand(d, k)  # W_k
    P_v = torch.rand(d, v)  # W_v
    P_o = torch.rand(h, d, v)  # W_o

    q = torch.einsum("bd,hdk->bhk", X, P_q)
    K = torch.concat([prev_K, torch.einsum("bd,dk->bk", M, P_k).unsqueeze(1)], axis=1)
    V = torch.concat([prev_V, torch.einsum("bd,dv->bv", M, P_v).unsqueeze(1)], axis=1)
    logits = torch.einsum("bhk,bmk->bhm", q, K)
    weights = torch.softmax(logits, dim=-1)
    O = torch.einsum("bhm,bmv->bhv", weights, V)
    y = torch.einsum("bhv,hdv->bd", O, P_o)
    return y, K, V


if __name__ == "__main__":
    MultiheadAttentionBatched()
    MultiheadSelfAttentionIncremental()
    MultiquerySelfAttentionIncremental()
