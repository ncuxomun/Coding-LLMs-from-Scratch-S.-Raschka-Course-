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
