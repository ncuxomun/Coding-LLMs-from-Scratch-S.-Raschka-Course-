#%%
# Coding Attention Mechanisms
## Attending to different parts of hte input with self-attention

### A simple self-attention mechanism without trainable weights

# %%
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
# %%
input_query = inputs[1]
print(input_query)
# %%
query = inputs[1] # 2nd input token

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
  attn_scores_2[i] = torch.dot(x_i, query)
# %%
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# %%
query = inputs[1] # 2nd input token

context_vec_2 = torch.zeros(size=query.shape)
for i, x_i in enumerate(inputs):
  # Summation happens in vertical stack
  context_vec_2 += attn_weights_2[i] * x_i
  
print(context_vec_2)
# %%
#* 3.3.2 A simple self-attention mechanism without trainable weights
query = inputs[1]

attn_scores = torch.empty(size=(inputs.shape[0], inputs.shape[0]))

for i, x_i in enumerate(inputs):
  for j, x_j in enumerate(inputs):
    attn_scores[i, j] = torch.dot(x_i, x_j)
    
print(attn_scores)
# %%
attn_scores = inputs @ inputs.T # unnormalized weights
print(attn_scores)
# %%
attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)
# %%
# context matrix
all_context_vecs = attn_weights @ inputs

print(all_context_vecs)
# %%
#* 3.4.1 Computing the attention weights step by step
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2 #output size of the context vector, i.e, (1, d_out)

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
W_keys = torch.nn.Parameter(torch.rand(d_in, d_out))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

# %%
query_2 = x_2 @ W_query
keys = inputs @ W_keys
values = inputs @ W_value

# %%
keys_2 = keys[1]
attn_score_22 = torch.dot(query_2, keys_2)

# %%
attn_scores_2 = query_2 @ keys.T
# attn_scores_2
d_k = keys.shape[1]

attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print(attn_weights_2)

# %%
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
# %%
#* 3.4.2 Implementing a compact SelfAttention Class

class SelfAttentionV1(torch.nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    
    self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
    self.W_keys = torch.nn.Parameter(torch.rand(d_in, d_out))
    self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
    
  def forward(self, x):
    queries = x @ self.W_query
    keys = x @ self.W_keys
    values = x @ self.W_value
    
    d_k = keys.shape[1]
    
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim=-1)
    context_vec = attn_weights @ values
    
    return context_vec

# %%
torch.manual_seed(123)

sa_v1 = SelfAttentionV1(d_in=d_in, d_out=d_out)
sa_v1(inputs)
# %%
class SelfAttentionV2(torch.nn.Module):
  def __init__(self, d_in, d_out, qkv_bias=False):
    super().__init__()
    
    self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_keys = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    
  def forward(self, x):
    queries = self.W_query(x)
    keys = self.W_keys(x)
    values = self.W_value(x)
    
    d_k = keys.shape[1]
    
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim=-1)
    context_vec = attn_weights @ values
    
    return context_vec

# %%
torch.manual_seed(123)

sa_v2 = SelfAttentionV2(d_in=d_in, d_out=d_out)
sa_v2(inputs)
# %%
#* 3.5 Hiding future words with causal attention
##* 3.5.1 Applying a causal attention 
##* 3.5.2 Masking additional attention weights with dropout
##* Implementing a compact a causal attention
batch = torch.stack((inputs, inputs), dim=0)

class CausalAttention(torch.nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
    super().__init__()
    
    self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_keys = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout = torch.nn.Dropout(dropout)
    self.register_buffer(name="mask",
                         tensor=torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
  def forward(self, x):
    b, num_tokens, d_in = x.shape
    # batch, num_tokens, embedding dimension
    
    queries = self.W_query(x)
    keys = self.W_keys(x)
    values = self.W_value(x)
    
    attn_scores = queries @ keys.transpose(1, 2)
    attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    context_vec = attn_weights @ values
    
    return context_vec

# %%
torch.manual_seed(789)

context_length = batch.shape[1]
dropout = 0.0

ca = CausalAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=dropout)
ca(batch)
# %%
#* 3.6 Extending single-head attention to multi-head attention
##* Stacking multiple single-head attention layers
class MultiHeadAttentionWrapper(torch.nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False, num_heads=2):
    super().__init__()
    self.heads = torch.nn.ModuleList(
      [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
    )
  
  def forward(self, x):
    return torch.cat([head(x) for head in self.heads], dim=-1)

# %%
torch.manual_seed(123)

context_length = batch.shape[1]
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0)
# %%
print(mha(batch))

# %%
##* 3.6.2 Implementing multi-head attention with weight splits
## More efficient implementation
class MultiHeadAttention(torch.nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    
    assert (d_in % num_heads == 0), "d_out must be divisible by num_heads"
    
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads # Reduce the project dim to match desired output dim
    
    self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    
    self.out_proj = torch.nn.Linear(d_out, d_out) # Linear layer to combine head outputs
    self.dropout = torch.nn.Dropout(dropout)
    
    self.register_buffer(
      name="mask",
      tensor=torch.triu(torch.ones(context_length, context_length), diagonal=1)
      )
      
  def forward(self, x):
    b, num_tokens, d_in = x.shape
    
    keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
    queries = self.W_query(x)
    values = self.W_value(x)
    
    # We implicitly split the matrix by adding a `num_heads` dimension
    # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    
    # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)
    
    # Compute scaled dot-product attention (aka self-attention) w/ causal mask
    attn_scores = queries @ keys.transpose(2, 3) # Dot product for each head
    
    # Original mask truncated to the number of tokens and converted to boolean
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    
    # Use the mask to fill attention scores
    attn_scores.masked_fill_(mask_bool, -torch.inf)
    
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    
    # Shape: (b, num_tokens, num_heads, head_dim)
    context_vec = (attn_weights @ values).transpose(1, 2)
    
    # Combine heads, where self.d_out = self.num_heads * self.head_dim
    context_vec = context_vec.reshape(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec) # optional projection
    
    return context_vec
    
# %%
torch.manual_seed(123)

inputs = torch.randn(size=(6, 4))

batch = torch.stack((inputs, inputs), dim=0)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
# %%
