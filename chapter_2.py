# Working with Text Data 
# Reference link - https://www.youtube.com/watch?v=341Rb8fJxY0&t=19s

#%%
#* 2.2 Tokenizing text
import pathlib as pb

file_path = pb.Path("datasets", "the-verdict.txt")

with open(file=file_path, encoding="utf-8") as f:
    raw_text = f.read()

# %%
# Doing some tokenization with Regular Expression
# 'Simplest' way to tokenize
import re

text = "Hello, world! This is a test."
result = re.split(pattern=r"(\s)", string=text)
print(result)

# more sophisticated splitting
result_2 = re.split(r"([,.?!]|\s)", text)
print(result_2)
# %%
# Tokenization for our text
result_raw = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result_raw = [item.strip() for item in result_raw if item.strip()]
preprocessed = result_raw
print(result_raw)
# %%
#* 2.3 Converting tokens into token IDs
all_words = sorted(set(preprocessed))
vocab_size = len(all_words) # of unique words we have

# Let's build vocabulary
vocab = {token: integer for integer, token in enumerate(iterable=all_words)}
# %%
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
# %%
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)

# Decode (from ids) to text
print(tokenizer.decode(ids))
# %%
#* 2.4 Adding special context tokens
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token: integer for integer, token in enumerate(iterable=all_tokens)}
# %%
# Modifying tokenizer for the extended vocab
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.decode(tokenizer.encode("Hello")))
# %%
#* Testing special tokens
text = "Hello, do you like tea? Is this__ a test?"

tokenizer.encode(text=text)
# %%
#* 2.5 Byte pair encoding
import tiktoken

tokenizer = tiktoken.get_encoding(encoding_name="gpt2")
# %%
print(tokenizer.encode(text="Hello world"))
tokenizer.decode(tokens=tokenizer.encode(text="Hello world"))
# %%
text = (
    "Hello, do you like marmalade? <|endoftext|> in snlit terraces..."
)
tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# %%
#* 2.6 Data sampling with a sliding window
file_path = pb.Path("datasets", "the-verdict.txt")

with open(file=file_path, encoding="utf-8") as f:
    raw_text = f.read()
    
enc_text = tokenizer.encode(raw_text)
# %%
enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")
# %%
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    
    print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))
# %%
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        # Use a sliding window to chunk the book into overlapping sequences of max length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# %%
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader

# %%
data_loader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=4, shuffle=False)

data_iter = iter(data_loader)
first_batch = next(data_iter)

print(first_batch)
# %%
#* 2.7 Creating token embeddings
input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=output_dim)
# %%
embedding_layer(torch.tensor(input_ids))
# %%
#* 2.8 Encoding word positions
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# %%
max_length = 4
data_loader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                                   stride=max_length, shuffle=False)

data_iter = iter(data_loader)
inputs, targets = next(data_iter)

# %%
token_embeddings = token_embedding_layer(inputs)
# %%
# Positional embedding to help with word positions in a sentence
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# %%
pos_embedding = pos_embedding_layer(torch.arange(context_length))
# %%
# now, input embedding vector - combination of words + positional embedding
input_embedding = token_embeddings + pos_embedding
# %%
