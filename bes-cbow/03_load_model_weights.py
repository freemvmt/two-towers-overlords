### This file serves as an example of how to load embedding model weights from a checkpoint.
import torch
from model import CBOW
import pickle 
from evaluate import topk
import numpy as np

# Paths to the vocabulary files
words_to_ids_path = './data/tkn_words_to_ids.pkl'
ids_to_words_path = './data/tkn_ids_to_words.pkl'

words_to_ids = pickle.load(open(words_to_ids_path, 'rb'))
ids_to_words = pickle.load(open(ids_to_words_path, 'rb'))

### set the vocabulary size and embedding dimension based on your training setup
# vocab_size = 63642  # Example vocabulary size (from bes CBOW training)
vocab_size = len(words_to_ids)  # Best to use the length of the words_to_ids dictionary
example_embedding_dim = 128  # Example embedding dimension (from bes CBOW training)

## Load the model weights from a checkpoint
cbow = CBOW(vocab_size, example_embedding_dim)
checkpoint_path = './checkpoints/bes-basic-cbow.cbow.pth'
cbow.load_state_dict(torch.load(checkpoint_path))
cbow.eval()  # Set the model to evaluation mode


### extract and use the embedding layer
embeddings = cbow.emb.weight.data  # Get the embedding weights
print(f"Loaded embeddings shape: {embeddings.shape}")  # Should be (vocab_size, embedding_dim)
# print an example embedding for a specific word
word_index = 5234  # Example index for the word 'anarchism'
print(f"Embedding for word {ids_to_words[word_index]} index {word_index}: {embeddings[word_index]}")  # Convert to numpy for better readability

### Example usage of the topk function to find similar words
topk(cbow, vocab='botswana')  # Replace 'computer' with any word you want to find similar words for