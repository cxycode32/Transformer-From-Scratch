import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import IWSLT2016
from torchtext.vocab import build_vocab_from_iterator
import os
import subprocess
import spacy
from spacy_download import load_spacy
from functools import partial
import xml.etree.ElementTree as ET
import config


def install_spacy_model(model_name):
    """
    Installs a missing spaCy model automatically if it's not installed.

    Args:
        model_name (str): The name of the spaCy model to install.
    """
    print(f"Installing missing Spacy model: {model_name}...")
    try:
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        print(f"Successfully installed {model_name}")
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Failed to install {model_name}. Please install manually: python -m spacy download {model_name}")
    
    
def load_tokenizers(src_lang, trg_lang):
    """
    Loads tokenizers for the specified source and target languages.

    Args:
        src_lang (str): Source language code.
        trg_lang (str): Target language code.

    Returns:
        tuple: Tokenizer functions for the source and target languages.
    """
    spacy_models = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "cs": "cs_core_news_sm",
        "ar": "ar_core_news_sm",
    }

    tokenizers = {}
    
    def tokenize_text(text, spacy_model):
        return [tok.text.lower() for tok in spacy_model.tokenizer(text)]

    for lang in [src_lang, trg_lang]:
        model_name = spacy_models.get(lang)
        if model_name is None:
            raise ValueError(f"Unsupported language: {lang}")

        try:
            spacy_model = load_spacy(model_name)
        except OSError:
            install_spacy_model(model_name)
            spacy_model = spacy.load(model_name)

        tokenizers[lang] = partial(tokenize_text, spacy_model=spacy_model)

    return tokenizers[src_lang], tokenizers[trg_lang]


def build_vocab(dataset, tokenizer, min_freq, max_size):
    """
    Builds a vocabulary from a dataset.

    Args:
        dataset (iterable): Dataset of sentence pairs.
        tokenizer (callable): Tokenizer function.
        min_freq (int): Minimum frequency for token inclusion.
        max_size (int): Maximum vocabulary size.

    Returns:
        Vocab: Constructed vocabulary.
    """
    def yield_tokens(data):
        for (src, trg) in data:
            yield tokenizer(src)
            yield tokenizer(trg)

    vocab = build_vocab_from_iterator(
        yield_tokens(dataset),
        min_freq=min_freq,
        max_tokens=max_size,
        specials=["<pad>", "<sos>", "<eos>", "<unk>"]
    )

    vocab.set_default_index(vocab["<unk>"])

    return vocab


def create_decode_function(vocab):
    """Creates a decode function to convert token indices to text."""
    index_to_word = {idx: word for word, idx in vocab.get_stoi().items()}  # Reverse vocab mapping

    def decode(indices):
        """Converts token indices back to a readable sentence."""
        return " ".join([index_to_word[idx] for idx in indices if index_to_word[idx] not in ["<pad>", "<sos>", "<eos>", "<unk>"]])

    return decode


def load_dataset(src_lang, trg_lang, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE):
    """
    Loads the IWSLT2016 dataset using torchtext.

    Args:
        src_lang (str): Source language.
        trg_lang (str): Target language.
        min_freq (int): Minimum frequency for vocab tokens.
        max_size (int): Maximum vocab size.

    Returns:
        tuple: Data splits, vocabularies, and tokenizers.
    """
    train_iter, valid_iter, test_iter = IWSLT2016(language_pair=(src_lang, trg_lang))

    src_tokenizer, trg_tokenizer = load_tokenizers(src_lang, trg_lang)
    
    train_data, valid_data, test_data = list(train_iter), list(valid_iter), list(test_iter)

    src_vocab = build_vocab(train_data, src_tokenizer, min_freq=min_freq, max_size=max_size)
    trg_vocab = build_vocab(train_data, trg_tokenizer, min_freq=min_freq, max_size=max_size)

    return train_data, valid_data, test_data, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer


def load_dataset_fallback(src_lang, trg_lang, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE, dir=config.DATASET_DIR):
    """
    Loads a locally stored IWSLT2016 dataset.

    Args:
        src_lang (str): Source language.
        trg_lang (str): Target language.
        min_freq (int): Minimum token frequency for vocab.
        max_size (int): Maximum vocab size.
        dir (str): Dataset directory.

    Returns:
        tuple:
            - train_data (list of tuples): Training data pairs (source sentence, target sentence).
            - valid_data (list of tuples): Validation data pairs (source sentence, target sentence).
            - test_data (list of tuples): Test data pairs (source sentence, target sentence).
            - src_vocab (Vocab): Vocabulary for the source language.
            - trg_vocab (Vocab): Vocabulary for the target language.
            - src_tokenizer (callable): Tokenizer function for the source language.
            - trg_tokenizer (callable): Tokenizer function for the target language.

    Raises:
        FileNotFoundError: If the dataset directory or expected dataset files are missing.
    """
    dataset_path = os.path.join(dir, f"{src_lang}-{trg_lang}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    def load_text_data(src_path, trg_path):
        """
        Loads parallel text data from plain text files.

        Args:
            src_path (str): Path to the source language text file.
            trg_path (str): Path to the target language text file.

        Returns:
            list of tuples: Each tuple contains (source sentence, target sentence).

        Raises:
            FileNotFoundError: If any of the dataset files are missing.
        """
        if not os.path.exists(src_path) or not os.path.exists(trg_path):
            raise FileNotFoundError(f"Missing dataset files: {src_path}, {trg_path}")

        def is_metadata(line):
            return line.startswith("<") and not line.startswith("<seg")

        src_sentences, trg_sentences = [], []

        with open(src_path, "r", encoding="utf-8") as src_file, open(trg_path, "r", encoding="utf-8") as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                if not is_metadata(src_line) and not is_metadata(trg_line):
                    src_sentences.append(src_line.strip())
                    trg_sentences.append(trg_line.strip())

        return list(zip(src_sentences, trg_sentences))
    
    def load_xml_data(src_path, trg_path):
        """
        Extracts sentence pairs from XML dataset files.

        Args:
            src_path (str): Path to the source language XML file.
            trg_path (str): Path to the target language XML file.

        Returns:
            list of tuples: Each tuple contains (source sentence, target sentence).

        Raises:
            FileNotFoundError: If any of the XML dataset files are missing.
            ValueError: If the XML file does not contain expected <seg> elements.
        """
        if not os.path.exists(src_path) or not os.path.exists(trg_path):
            raise FileNotFoundError(f"Missing dataset files: {src_path}, {trg_path}")

        def extract_text_from_xml(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            segments = root.findall(".//seg")

            if not segments:
                raise ValueError(f"XML file {xml_path} does not contain <seg> elements.")

            return [seg.text.strip() for seg in segments if seg.text]

        src_sentences = extract_text_from_xml(src_path)
        trg_sentences = extract_text_from_xml(trg_path)

        return list(zip(src_sentences, trg_sentences))

    train_src_path = os.path.join(dataset_path, f"train.tags.{src_lang}-{trg_lang}.{src_lang}")
    train_trg_path = os.path.join(dataset_path, f"train.tags.{src_lang}-{trg_lang}.{trg_lang}")

    valid_src_path = os.path.join(dataset_path, f"IWSLT16.TED.dev2010.{src_lang}-{trg_lang}.{src_lang}.xml")
    valid_trg_path = os.path.join(dataset_path, f"IWSLT16.TED.dev2010.{src_lang}-{trg_lang}.{trg_lang}.xml")

    test_src_path = os.path.join(dataset_path, f"IWSLT16.TED.tst2010.{src_lang}-{trg_lang}.{src_lang}.xml")
    test_trg_path = os.path.join(dataset_path, f"IWSLT16.TED.tst2010.{src_lang}-{trg_lang}.{trg_lang}.xml")

    train_data = load_text_data(train_src_path, train_trg_path)
    valid_data = load_xml_data(valid_src_path, valid_trg_path)
    test_data = load_xml_data(test_src_path, test_trg_path)

    src_tokenizer, trg_tokenizer = load_tokenizers(src_lang, trg_lang)
    src_vocab = build_vocab(train_data, src_tokenizer, min_freq=min_freq, max_size=max_size)
    trg_vocab = build_vocab(train_data, trg_tokenizer, min_freq=min_freq, max_size=max_size)

    src_tokenizer.decode = create_decode_function(src_vocab)
    trg_tokenizer.decode = create_decode_function(trg_vocab)

    return train_data, valid_data, test_data, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer


def collate_fn(batch, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, pad_idx):
    """
    Custom collate function for dynamically padding sequences within a batch.
    
    Args:
        batch (list of tuples): Each tuple contains (src_sentence, trg_sentence).
        src_vocab (Vocab): Vocabulary object for source language.
        trg_vocab (Vocab): Vocabulary object for target language.
        src_tokenizer (callable): Tokenizer function for source language.
        trg_tokenizer (callable): Tokenizer function for target language.
        pad_idx (int): Padding index.

    Returns:
        torch.Tensor: Padded source and target tensors.
    """
    src_batch, trg_batch = [], []

    for src_sentence, trg_sentence in batch:
        src_indices = [src_vocab[token] for token in src_tokenizer(src_sentence)]
        trg_indices = [trg_vocab[token] for token in trg_tokenizer(trg_sentence)]

        src_indices = [src_vocab["<sos>"]] + src_indices + [src_vocab["<eos>"]]
        trg_indices = [trg_vocab["<sos>"]] + trg_indices + [trg_vocab["<eos>"]]

        src_batch.append(torch.tensor(src_indices, dtype=torch.long))
        trg_batch.append(torch.tensor(trg_indices, dtype=torch.long))

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)  # Shape: (batch_size, max_len)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)  # Shape: (batch_size, max_len)

    return src_padded, trg_padded


def create_dataloaders(train_data, valid_data, test_data, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, batch_size=config.BATCH_SIZE):
    """
    Create PyTorch dataloaders for training, validation, and testing.

    Args:
        train_data, valid_data, test_data: Dataset splits.
        src_vocab, trg_vocab: Vocab objects for source and target languages.
        src_tokenizer, trg_tokenizer: Tokenizer functions.
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle training data.

    Returns:
        Tuple of DataLoader objects.
    """
    pad_idx = src_vocab["<pad>"]

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, pad_idx)
    )

    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, pad_idx)
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, pad_idx)
    )

    return train_loader, valid_loader, test_loader