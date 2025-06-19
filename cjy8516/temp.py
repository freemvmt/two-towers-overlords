import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import random
from typing import List, Tuple, Dict
import pickle
import json
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
import faiss

tokenizer_name='bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
max_seq_length = 128


queries = {}
documents = {}
qrels = {}
query_ids = df['query_id']
query_text = df['query']

# Store query
queries = dict(zip(query_ids, query_text))

for i, passage in enumerate(df['passages']):
  for j, text in enumerate(passage['passage_text']):
    id = df.loc[i, 'query_id']
    doc_id = f"{id}_{j}"
    documents[doc_id] = passage['passage_text'][j]
    if id not in qrels:
      qrels[id] = []
      qrels[id].append(doc_id)

all_doc_ids = list(documents.keys())
triplets = []
query_id_set = set(query_ids)
for query_id in query_ids:
  query_text = queries[query_id]
  rest_of_ids = list(query_id_set - {query_id})
  neg_doc_query_id = random.choice(rest_of_ids)
  relevant_doc_ids = qrels[query_id]
  pos_doc_id = random.choice(relevant_doc_ids)
  neg_doc_id = random.choice(qrels[neg_doc_query_id])

  # negative_pool = list(set(all_doc_ids) - set(relevant_doc_ids))
  # neg_doc_id = random.choice(negative_pool)

  pos_doc_text = documents[pos_doc_id]
  neg_doc_text = documents[neg_doc_id]
  triplets.append({
                      'query': query_text,
                      'positive_doc': pos_doc_text,
                      'negative_doc': neg_doc_text,
                      'query_id': query_id,
                      'pos_doc_id': pos_doc_id,
                      'neg_doc_id': neg_doc_id
                  })

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenized_triplets = []
for triplet in tqdm(triplets, desc="Tokenizing"):
            # Tokenize query
            query_tokens = tokenizer(
                triplet['query'],
                padding='max_length',
                truncation=True,
                max_length=max_seq_length,
                return_tensors='pt'
            )

            # Tokenize positive document
            pos_doc_tokens = tokenizer(
                triplet['positive_doc'],
                padding='max_length',
                truncation=True,
                max_length=max_seq_length,
                return_tensors='pt'
            )

            # Tokenize negative document
            neg_doc_tokens = tokenizer(
                triplet['negative_doc'],
                padding='max_length',
                truncation=True,
                max_length=max_seq_length,
                return_tensors='pt'
            )

            tokenized_triplets.append({
                'query_input_ids': query_tokens['input_ids'].squeeze(),
                'query_attention_mask': query_tokens['attention_mask'].squeeze(),
                'pos_doc_input_ids': pos_doc_tokens['input_ids'].squeeze(),
                'pos_doc_attention_mask': pos_doc_tokens['attention_mask'].squeeze(),
                'neg_doc_input_ids': neg_doc_tokens['input_ids'].squeeze(),
                'neg_doc_attention_mask': neg_doc_tokens['attention_mask'].squeeze(),
            })

class TripletDataset(Dataset):
    """
    PyTorch Dataset class for handling our triplet data
    This makes it easy to load data in batches during training
    """

    def __init__(self, tokenized_triplets):
        self.triplets = tokenized_triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

class AveragePoolingEncoder(nn.Module):
    """
    Average Pooling Encoder

    This encoder processes sequential text data by using simple average pooling
    over token embeddings. It's much simpler and more efficient than LSTM-based
    approaches while still producing meaningful representations.

    This encoder takes tokenized text and converts it into a fixed-size vector
    representation by averaging the embeddings of all tokens in the sequence.
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=None):
        """
        Initialize the encoder

        Args:
            vocab_size: Size of the vocabulary (number of unique tokens)
            embedding_dim: Dimension of token embeddings
            hidden_dim: Output dimension after projection
            num_layers: Not used in this implementation (kept for compatibility)
        """
        super(AveragePoolingEncoder, self).__init__()

        # Embedding layer: converts token IDs to dense vectors
        # This is like a lookup table where each token gets a learnable vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Optional projection layer to transform averaged embeddings
        # If hidden_dim != embedding_dim, we project to the desired size
        if hidden_dim != embedding_dim:
            self.projection = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.projection = None

        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.1)

        # Store hidden_dim for reference
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the encoder

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Mask to ignore padding tokens [batch_size, seq_length]

        Returns:
            encoding: Fixed-size vector representation [batch_size, hidden_dim]
        """
        # Get embeddings for input tokens
        embeddings = self.embedding(input_ids)  # [batch_size, seq_length, embedding_dim]

        # Apply dropout to embeddings
        embeddings = self.dropout(embeddings)

        # Apply attention mask to ignore padding tokens
        attention_mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_length, 1]
        masked_embeddings = embeddings * attention_mask  # Zero out padded positions

        # Average pooling: average over sequence length, ignoring padded tokens
        seq_lengths = attention_mask.sum(dim=1)  # [batch_size, 1]
        # Avoid division by zero for empty sequences
        seq_lengths = seq_lengths.clamp(min=1.0)
        pooled = masked_embeddings.sum(dim=1) / seq_lengths  # [batch_size, embedding_dim]

        # Optional projection to desired hidden dimension
        if self.projection is not None:
            encoding = self.projection(pooled)  # [batch_size, hidden_dim]
        else:
            encoding = pooled  # [batch_size, embedding_dim]

        return encoding

class TwoTowerModel(nn.Module):
    """
    Two-Tower Neural Network Architecture

    This consists of two separate encoders:
    1. Query Tower: Encodes user queries
    2. Document Tower: Encodes documents

    The idea is that queries and documents have different characteristics,
    so they might benefit from separate encoding networks.
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(TwoTowerModel, self).__init__()

        # Query encoder (Tower 1)
        self.query_encoder = AveragePoolingEncoder(vocab_size, embedding_dim, hidden_dim, num_layers)

        # Document encoder (Tower 2)
        self.document_encoder = AveragePoolingEncoder(vocab_size, embedding_dim, hidden_dim, num_layers)

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        """
        Forward pass through both towers

        Returns:
            query_encoding: Encoded query vector
            doc_encoding: Encoded document vector
        """
        # Encode query using query tower
        query_encoding = self.query_encoder(query_input_ids, query_attention_mask)

        # Encode document using document tower
        doc_encoding = self.document_encoder(doc_input_ids, doc_attention_mask)

        return query_encoding, doc_encoding

class TripletLoss(nn.Module):
    """
    Triplet Loss Function

    This loss function trains the model to:
    - Make query-relevant_document distance small
    - Make query-irrelevant_document distance large

    The loss is: max(0, distance(q,pos) - distance(q,neg) + margin)
    """

    def __init__(self, margin=1.0, distance_function='cosine'):
        """
        Initialize triplet loss

        Args:
            margin: How much separation we want between positive and negative examples
            distance_function: 'cosine' or 'euclidean'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_function = distance_function

    def compute_distance(self, x, y):
        """
        Compute distance between two vectors
        """
        if self.distance_function == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            # Cosine similarity = dot_product / (norm_x * norm_y)
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
            cosine_similarity = torch.sum(x_norm * y_norm, dim=1)
            return 1 - cosine_similarity

        elif self.distance_function == 'euclidean':
            # Euclidean distance = sqrt(sum((x - y)^2))
            return torch.norm(x - y, p=2, dim=1)

        else:
            raise ValueError(f"Unknown distance function: {self.distance_function}")

    def forward(self, query_encoding, pos_doc_encoding, neg_doc_encoding):
        """
        Compute triplet loss

        Args:
            query_encoding: Encoded query vectors [batch_size, hidden_dim]
            pos_doc_encoding: Encoded positive document vectors [batch_size, hidden_dim]
            neg_doc_encoding: Encoded negative document vectors [batch_size, hidden_dim]

        Returns:
            loss: Triplet loss value
        """
        # Distance between query and positive document (should be small)
        pos_distance = self.compute_distance(query_encoding, pos_doc_encoding)

        # Distance between query and negative document (should be large)
        neg_distance = self.compute_distance(query_encoding, neg_doc_encoding)

        # Triplet loss: we want pos_distance < neg_distance by at least margin
        # Loss = max(0, pos_distance - neg_distance + margin)
        loss = torch.relu(pos_distance - neg_distance + self.margin)

        return loss.mean()

class DocumentRetriever:
    """
    This class handles the inference/retrieval process

    During inference:
    1. Pre-encode all documents once
    2. For each query, encode it and find most similar documents
    3. Return top-k most relevant documents
    """

    def __init__(self, model, tokenizer, max_seq_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.document_embeddings = None
        self.document_texts = []
        self.document_ids = []

        # For efficient similarity search, we can use FAISS
        # This is similar to what ChromaDB provides
        self.index = None

    def encode_documents(self, documents: Dict[str, str]):
        """
        Pre-encode all documents for fast retrieval

        Args:
            documents: Dictionary of {doc_id: doc_text}
        """
        # logger.info("Encoding documents for retrieval...")

        self.document_texts = []
        self.document_ids = []
        embeddings = []

        # Process documents in batches for efficiency
        batch_size = 32
        doc_items = list(documents.items())

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(doc_items), batch_size), desc="Encoding documents"):
                batch_items = doc_items[i:i+batch_size]
                batch_texts = [item[1] for item in batch_items]
                batch_ids = [item[0] for item in batch_items]

                # Tokenize batch
                tokenized = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors='pt'
                )

                # Encode documents
                _, doc_embeddings = self.model(
                    torch.zeros_like(tokenized['input_ids']),  # Dummy query input
                    torch.zeros_like(tokenized['attention_mask']),
                    tokenized['input_ids'],
                    tokenized['attention_mask']
                )

                embeddings.append(doc_embeddings.cpu().numpy())
                self.document_texts.extend(batch_texts)
                self.document_ids.extend(batch_ids)

        # Combine all embeddings
        self.document_embeddings = np.vstack(embeddings)

        # Build FAISS index for fast similarity search
        # This is an alternative to ChromaDB for vector similarity search
        embedding_dim = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (for cosine similarity)

        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.document_embeddings / np.linalg.norm(
            self.document_embeddings, axis=1, keepdims=True
        )
        self.index.add(normalized_embeddings.astype('float32'))

        # logger.info(f"Encoded {len(self.document_ids)} documents")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieve top-k most relevant documents for a query

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of (doc_id, doc_text, similarity_score) tuples
        """
        if self.document_embeddings is None:
            raise ValueError("Documents must be encoded first. Call encode_documents().")

        # Tokenize query
        tokenized_query = self.tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

        # Encode query
        self.model.eval()
        with torch.no_grad():
            query_embedding, _ = self.model(
                tokenized_query['input_ids'],
                tokenized_query['attention_mask'],
                torch.zeros_like(tokenized_query['input_ids']),  # Dummy document input
                torch.zeros_like(tokenized_query['attention_mask'])
            )

        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.cpu().numpy()
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search for similar documents using FAISS
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Format results
        results = []
        for i, (similarity, doc_idx) in enumerate(zip(similarities[0], indices[0])):
            doc_id = self.document_ids[doc_idx]
            doc_text = self.document_texts[doc_idx]
            results.append((doc_id, doc_text, float(similarity)))

        return results

def train_model(model, train_loader, num_epochs=10, learning_rate=1e-3):
    """
    Training function for the Two-Tower model

    Args:
        model: TwoTowerModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    # Set up loss function and optimizer
    criterion = TripletLoss(margin=1.0, distance_function='cosine')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()

            # Forward pass for positive documents
            query_enc, pos_doc_enc = model(
                batch['query_input_ids'],
                batch['query_attention_mask'],
                batch['pos_doc_input_ids'],
                batch['pos_doc_attention_mask']
            )

            # Forward pass for negative documents
            _, neg_doc_enc = model(
                batch['query_input_ids'],
                batch['query_attention_mask'],
                batch['neg_doc_input_ids'],
                batch['neg_doc_attention_mask']
            )

            # Compute loss
            loss = criterion(query_enc, pos_doc_enc, neg_doc_enc)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # Validation phase

        model.eval()
        """
val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Same forward pass as training
                query_enc, pos_doc_enc = model(
                    batch['query_input_ids'],
                    batch['query_attention_mask'],
                    batch['pos_doc_input_ids'],
                    batch['pos_doc_attention_mask']
                )

                _, neg_doc_enc = model(
                    batch['query_input_ids'],
                    batch['query_attention_mask'],
                    batch['neg_doc_input_ids'],
                    batch['neg_doc_attention_mask']
                )

                loss = criterion(query_enc, pos_doc_enc, neg_doc_enc)
                val_loss += loss.item()
                val_batches += 1
        """


        # Print epoch results
        avg_train_loss = train_loss / train_batches
        # avg_val_loss = val_loss / val_batches

        # logger.info(f"Epoch {epoch+1}/{num_epochs}")
        # logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        # logger.info(f"  Val Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            # 'val_loss': avg_val_loss,
        }, f'model_checkpoint_epoch_{epoch+1}.pth')

vocab_size = tokenizer.vocab_size
model = TwoTowerModel(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
train_triplets = tokenized_triplets
train_dataset = TripletDataset(train_triplets)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_model(model, train_loader, num_epochs=5, learning_rate=1e-3)
retriever = DocumentRetriever(model, tokenizer)
retriever.encode_documents(documents)





