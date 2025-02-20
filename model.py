import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Implements the scaled dot-product self-attention mechanism, which allows the model to focus on different parts
    of the input sequence when generating an output.

    Args:
        embedding_size (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads in multi-head self-attention.
    """
    def __init__(self, embedding_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        assert (
            self.head_dim * num_heads == embedding_size
        ), "Embedding size needs to be divisible by the number of heads."

        self.values = nn.Linear(embedding_size, embedding_size)
        self.keys = nn.Linear(embedding_size, embedding_size)
        self.queries = nn.Linear(embedding_size, embedding_size)
        self.fc_out = nn.Linear(embedding_size, embedding_size)

    def forward(self, values, keys, query, mask):
        """
        Computes the self-attention mechanism over input sequences.

        Args:
            values (Tensor): The value tensor of shape (batch_size, value_len, embed_size).
            keys (Tensor): The key tensor of shape (batch_size, key_len, embed_size).
            query (Tensor): The query tensor of shape (batch_size, query_len, embed_size).
            mask (Tensor, optional): Mask to prevent attention to certain positions (shape: (batch_size, 1, 1, seq_len)).

        Returns:
            Tensor: The output after applying self-attention, with shape (batch_size, query_len, embed_size).
        """
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces for multi-head attention
        queries = self.queries(query)  # (N, query_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        values = self.values(values)  # (N, value_len, embed_size)

        # Split the embedding into self.heads different pieces
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)

        # Compute scaled dot-product attention scores (query * keys)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply softmax normalization
        # attention: (N, num_heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embedding_size ** (1 / 2)), dim=3)

        # Weighted sum of values based on attention scores
        output = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )

        # Final linear transformation
        output = self.fc_out(output)

        return output


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of a self-attention layer and a feed-forward network.

    Args:
        embedding_size (int): The size of the embedding vector.
        num_heads (int): Number of attention heads.
        forward_expansion (int): Expansion factor for the feed-forward network.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, embedding_size, num_heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, num_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """
        Forward pass through the Transformer block.

        Args:
            value (Tensor): Value tensor for attention computation.
            key (Tensor): Key tensor for attention computation.
            query (Tensor): Query tensor for attention computation.
            mask (Tensor): Mask to control attention.

        Returns:
            Tensor: Output tensor of the Transformer block.
        """
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        output = self.dropout(self.norm2(forward + x))
        return output


class Encoder(nn.Module):
    """
    Encoder module of the Transformer model. It processes the input sequence 
    through multiple Transformer blocks to generate contextualized representations.

    Args:
        src_vocab_size (int): Vocabulary size of the source language.
        embedding_size (int): Dimensionality of word embeddings.
        num_layers (int): Number of Transformer blocks (stacked layers).
        num_heads (int): Number of attention heads in multi-head self-attention.
        forward_expansion (int): Expansion factor for the feed-forward network.
        dropout (float): Dropout probability to prevent overfitting.
        max_length (int): Maximum sequence length for positional encoding.
        device (torch.device): Device to run computations on ('cuda' or 'cpu').
    """
    def __init__(
        self,
        src_vocab_size,
        embedding_size,
        num_layers,
        num_heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)

        # Stack multiple Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size,
                    num_heads,
                    forward_expansion=forward_expansion,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len), where each value represents a token index.
            mask (Tensor): Attention mask to prevent attending to padding tokens.

        Returns:
            Tensor: Encoder output of shape (batch_size, seq_len, embedding_size) 
                    after being processed by multiple Transformer layers.
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        output = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # The query, key, and value are the same for Encoder because
        # they come from the same input (enc_src).
        for layer in self.layers:
            output = layer(output, output, output, mask)
            
        return output


class DecoderBlock(nn.Module):
    """
    A single block of the Transformer decoder. It applies self-attention 
    on the target sequence and cross-attention on the encoder’s output.

    Args:
        embedding_size (int): Dimensionality of embeddings.
        num_heads (int): Number of attention heads.
        forward_expansion (int): Expansion factor for the feed-forward network.
        dropout (float): Dropout probability for regularization.
    """
    def __init__(self, embedding_size, num_heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embedding_size)
        self.attention = SelfAttention(embedding_size, num_heads=num_heads)
        self.transformer_block = TransformerBlock(
            embedding_size, num_heads, forward_expansion, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """
        Forward pass of a single decoder block.

        Args:
            x (Tensor): Target sequence embeddings of shape (batch_size, trg_seq_len, embedding_size).
            value (Tensor): Encoder output, used for cross-attention.
            key (Tensor): Same as `value`, used for cross-attention.
            src_mask (Tensor): Mask for the source sequence to prevent attending to padding tokens.
            trg_mask (Tensor): Mask for the target sequence to enforce causal (left-to-right) attention.

        Returns:
            Tensor: Processed output of the decoder block.
        """
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        output = self.transformer_block(value, key, query, src_mask)
        return output


class Decoder(nn.Module):
    """
    Processes the target sequence using multiple decoder blocks 
    and generates predictions.

    Args:
        trg_vocab_size (int): Size of the target vocabulary.
        embedding_size (int): Token embedding dimension.
        num_layers (int): Number of Transformer blocks in the decoder.
        num_heads (int): Number of attention heads in multi-head attention.
        forward_expansion (int): Expansion factor for the feed-forward network.
        dropout (float): Dropout rate for regularization.
        max_length (int): Maximum sequence length for positional encoding.
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
    """
    def __init__(
        self,
        trg_vocab_size,
        embedding_size,
        num_layers,
        num_heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)

        # Stack decoder blocks
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embedding_size, num_heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """
        Forward pass through the decoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, trg_seq_len).
            enc_out (Tensor): Output from the encoder.
            src_mask (Tensor): Mask for the source sequence.
            trg_mask (Tensor): Mask for the target sequence.

        Returns:
            Tensor: Decoder output after processing through decoder layers and final output layer.
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # For Decoder, the query comes from the trg_seq
        # The key and value come from the src_seq
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        output = self.fc_out(x)

        return output


class Transformer(nn.Module):
    """
    A full Transformer model consisting of:
    - An encoder that processes the source sequence.
    - A decoder that generates the target sequence.
    - A final linear layer for output predictions.

    Supports masking to handle padding and autoregressive decoding.

    Args:
        src_vocab_size (int): Source vocabulary size.
        trg_vocab_size (int): Target vocabulary size.
        src_pad_idx (int): Padding token index in the source vocabulary.
        trg_pad_idx (int): Padding token index in the target vocabulary.
        embedding_size (int): Dimension of token embeddings.
        num_layers (int): Number of layers in both encoder and decoder.
        num_heads (int): Number of attention heads in multi-head attention.
        forward_expansion (int): Expansion factor for feed-forward network.
        dropout (float): Dropout probability.
        max_length (int): Maximum sequence length.
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
    """
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embedding_size,
        num_layers,
        num_heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ):
        """
        Initializes the Transformer model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            trg_vocab_size (int): Size of the target vocabulary.
            src_pad_idx (int): Index of the padding token in the source vocabulary.
            trg_pad_idx (int): Index of the padding token in the target vocabulary.
            embedding_size (int): Dimension of the token embeddings.
            num_layers (int): Number of layers in both the encoder and decoder.
            num_heads (int): Number of attention heads in multi-head attention.
            forward_expansion (int): Expansion factor for the feed-forward network.
            dropout (float): Dropout probability for regularization.
            max_length (int): Maximum sequence length for positional encoding.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size,
            embedding_size,
            num_layers,
            num_heads,
            forward_expansion,
            dropout,
            max_length,
            device,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embedding_size,
            num_layers,
            num_heads,
            forward_expansion,
            dropout,
            max_length,
            device,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        Creates a mask for the source sequence to ignore padding tokens 
        during attention computations.

        Args:
            src (Tensor): Source sequence tensor of shape (batch_size, seq_length).

        Returns:
            Tensor: Mask of shape (batch_size, 1, 1, seq_length), where 
                    1 indicates valid tokens and 0 indicates padding.
        """
        src_mask = (src != self.src_pad_idx)  # (batch_size, seq_length)
        src_mask = src_mask.unsqueeze(1)  # (batch_size, 1, seq_length)
        src_mask = src_mask.unsqueeze(2)  # (batch_size, 1, 1, seq_length)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        """
        Creates a causal mask for the target sequence to prevent attention 
        to future tokens (autoregressive decoding).

        Args:
            trg (Tensor): Target sequence tensor of shape (batch_size, seq_length).

        Returns:
            Tensor: Lower triangular mask of shape (batch_size, 1, trg_length, trg_length),
                    where future tokens are masked.
        """
        N, trg_len = trg.shape  # N: batch_size, trg_len: seq_length
        trg_mask = torch.tril(torch.ones((trg_len, trg_len)))  # (trg_length, trg_length)
        trg_mask = trg_mask.expand(N, 1, trg_len, trg_len)  # (batch_size, 1, trg_length, trg_length)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        """
        Performs the forward pass of the Transformer model.

        Steps:
        1. Generate masks for the source and target sequences.
        2. Encode the source sequence.
        3. Decode the target sequence using the encoded representation.
        4. Return predicted token logits.

        Args:
            src (Tensor): Source sequence tensor of shape (batch_size, seq_length).
            trg (Tensor): Target sequence tensor of shape (batch_size, seq_length).

        Returns:
            Tensor: Predicted token logits of shape (batch_size, seq_length, trg_vocab_size).
        """
        src_mask = self.make_src_mask(src)  # (batch_size, 1, 1, seq_length)
        trg_mask = self.make_trg_mask(trg)  # (batch_size, 1, trg_length, trg_length)
        enc_src = self.encoder(src, src_mask)  # (batch_size, seq_length, embedding_size)
        output = self.decoder(trg, enc_src, src_mask, trg_mask)  # (batch_size, seq_length, trg_vocab_size)
        return output
