import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = []
    for q, k, v in zip(*qkv_heads):
        out_heads.append(attention(q, k, v, causal_mask))
        import pdb; pdb.set_trace()
    #out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]

def softmax_new(x):
    return softmax(x)

def layer_norm_new(x, g, b):
    return layer_norm(x, g, b)

def new_transformer(x, attn, ln_1, ln_2, mlp, n_head):
    """
    x: (n_seq, n_embed) \ (10, 768)
    """
    x_pre_mha = x

    c_attn = attn['c_attn']
    attn_c_proj = attn['c_proj']

    ln_1_b = ln_1['b']
    ln_1_g = ln_1['g']

    ln_2_b = ln_2['b']
    ln_2_g = ln_2['g']

    c_fc = mlp['c_fc']
    c_proj = mlp['c_proj']

    # Call layer norm on the input to transformer block,
    # and then layer norm on the input prior to FFN.

    # c_attn['b'] = (2304), which works out to (3, 768)
    # c_attn['w'] = (768, 2304)

    # c_fc['w'] = (3072, 768)
    # c_proj['w'] = (3072, 768)
    # c_proj['b'] = (768)

    x = layer_norm_new(x, ln_1_g, ln_1_b)

    res = []

    n_embd = c_attn['w'].shape[-1] # (2304)
    n_embd_per_head = n_embd // 12
    for i in range(n_head):
        start = i*n_embd_per_head
        end = (i+1)*n_embd_per_head

        qkv = x @ c_attn['w'][:, start:end] + c_attn['b'][start:end] # (10, 2304)
        q, k, v = np.split(qkv, 3, axis=-1)

        mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
        attn = softmax_new(q @ k.T / k.shape[-1] ** 0.5 + mask) @ v

        res.append(attn)

    x = np.stack(res, axis=0)
    x = np.transpose(x, [1, 0, 2])
    x = np.reshape(x, (x.shape[0], -1))
    # This is equivalent.
    #x = np.hstack(res)

    x = x @ attn_c_proj['w'] + attn_c_proj['b']

    x = x_pre_mha + x
    x_pre_ffn = x

    # FFN:
    x = layer_norm_new(x, ln_2_g, ln_2_b)
    x = gelu(x @ c_fc['w'] + c_fc['b'])
    x = x @ c_proj['w'] + c_proj['b']

    return x_pre_ffn + x



def gpt2_new(inputs, wte, wpe, blocks, ln_f, n_head): # [n_seq] -> [n_seq, n_vocab]
    # wte: (50257, 768)
    # wpe: (1024, 768)
    # inputs: List of integers of tokens.
    # blocks: (12) List of blocks with attn, ln_1, ln_2, mlp.
    # ln_f: b, g. 
    # n_head: number of heads (12)

    # (10, 768)
    seq_embedding = wte[inputs]
    pos_embedding = wpe[range(len(inputs))]

    x = seq_embedding + pos_embedding

    for block in blocks:
        attn = block['attn']
        ln_1 = block['ln_1']
        ln_2 = block['ln_2']
        mlp = block['mlp']
        x = new_transformer(x, attn, ln_1, ln_2, mlp, n_head)
        import pdb; pdb.set_trace()
    
    x = layer_norm_new(x, ln_f['g'], ln_f['b'])
    x = x @ wte.T
    return x




def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)
