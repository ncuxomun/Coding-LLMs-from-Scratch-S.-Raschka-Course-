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
#* Adding special context tokens
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
