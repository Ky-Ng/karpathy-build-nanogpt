import tiktoken
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
Differences from Transformer Paper
Layer norms go before MLPs and Layer norm at the end
Encoder only, no cross attention (special attention used in encoders)
- https://www.geeksforgeeks.org/cross-attention-mechanism-in-transformers/
"""


@dataclass
class GPTConfig:
    block_size: int = 1024  # Max Context length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768  # Hidden Size


class CausalSelfAttention(nn.Module):
    """
    Multi Headed Attention Mechanism
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # QKV Matrices
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output project (Mixing of all of the heads)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_hs = config.n_embd // config.n_head

        # Create a mask for the autoregressive attention, use .view(1,1,T,T) to broadcast to #batches/#heads later
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).view(
            1, 1, config.block_size, config.block_size)))

    def forward(self, x: torch.tensor):
        # B = Batch Size, T = sequence length, C = n_embd = "_" for now
        B, T, C = x.size()

        # Step 1) Generate the K,Q,V values (matrix format)
        """
        x = (B,T,C)
        c_attn = (C, 3*C)
        qkv = (B,T,C) x (C, 3*C) = (B,T,3*C)
        """
        qkv = self.c_attn(x)

        # in the T,3*C, we'll split the massive matrix into three separate TxC matrices
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Split each QKV matrix into its respective heads
        # First, split the TxC with vertical lines in the matrix denoting each of the heads: shape = (B, T, self.n_head, self.n_hs)
        # Then create shape (B, self.n_head, T, self.n_hs);
        # allow processing for each head in parallel with (T x self.n_hs); imagine self.n_head as a 3rd "depth" dimension
        q = q.view(B, T, self.n_head, self.n_hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.n_hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_hs).transpose(1, 2)

        # Step 2) Scaled Dot Product Attention
        """
        q = (B, self.n_head, T, self.n_hs)
        k.transpose(-2, -1) = (B, self.n_head, self.n_hs, T)
        attn = (B, self.n_head, T, T)
        - Note: each row, attn[i] in the attn matrix is how much token[i] should attend to all other token[j]
        - We'll apply this in the next part to each component of all v[j] in the self.n_embd
        """
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.n_hs))

        # Mask out attentions of future tokens (autoregressive)
        # self.bias[:, :, :T, :T], for every batch and head, mask out all tokens with 0 to -inf
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Apply softmax row-wise
        attn = F.softmax(attn, dim=-1)

        # Step 3) Reduce/Take weighted sum of the value vectors
        """
        attn = (B, self.n_head, T, T)
        v = (B, self.n_head, T, self.n_hs) 
        y = (B, self.n_head, T, self.n_hs) where each row y[i] corresponds to new rep of token[i]

        applies the weighting of the vectors to each component of the vectors
        """
        y = attn @ v

        # Step 4) Concatentate the vectors tip to tip
        # y = (B, T, self.n_head, self.n_hs) => (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 5) Final mixing
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    Feed Forward Network projecting to 4 * model dimension (config.n_embd)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # project to 4 * the model dimension in FFN

        # c stand for component (part of a nn.Module component rather than a block in the diagram)
        # fc = fully connected
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        # proj = project back down
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Normalization 1
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # MHA Attention
        self.attn = CausalSelfAttention(config)

        # Normalization 2
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Feed Forward Network
        self.mlp = MLP(config)

    def forward(self, x):
        # Residually apply the layer normalization to the input, then pass to attention
        x = x + self.attn(self.ln_1(x))

        # Residually apply the layer normalization to output of MHA, then pass to FFN
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # Embedding = wrapper for a tensor that you can index into the rows
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),

                # Index each layer from [0, config.n_layer); gray image in AIAYN
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),

                # special for GPT2
                ln_f=nn.LayerNorm(config.n_embd),

            )
        )

        # Language head at the end
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):

        # (B, T) batch size and token size
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence request of {T} larger than {self.config.block_size}"

        # Generate positional and token embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb  # Implicit broadcasting of pos_emb to every batch

        # Propogate input through each transformer block
        for block in self.transformer.h:
            x = block(x)

        # Run the final encoder's hidden representation through the layer norm and Language Head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Shape B,T,vocab_size
        return logits

    # Initialization code from Karpathy repo to test validity of our version

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            # 124M params
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            # 350M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # always 50257 for GPT model checkpoints
        config_args['vocab_size'] = 50257
        # always 1024 for GPT model checkpoints
        config_args['block_size'] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# model = GPT.from_pretrained("gpt2")
model = GPT(GPTConfig())
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device {device}")

model.eval()
model.to(device)

# TODO Review and understand more deeply
num_return_seq = 5
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello I am a language model, ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)  # (5xT)
x = tokens.to(device) # (B, T)

max_length = 30

torch.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # (B,T,vocab_size); grab only the probabilities of the next word
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, 1)

        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_seq):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print("> ", decoded)
