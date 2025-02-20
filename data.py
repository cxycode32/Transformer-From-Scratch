import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
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
    Installs a missing spaCy model if it's not already installed.

    This function attempts to install a specified spaCy model using subprocess. 
    If the installation fails, it raises a RuntimeError.

    Args:
        model_name (str): The name of the spaCy model to install (e.g., "en_core_web_sm").

    Raises:
        RuntimeError: If the model installation fails.
    """
    print(f"Installing missing Spacy model: {model_name}...")
    try:
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        print(f"Successfully installed {model_name}")
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Failed to install {model_name}. Please install manually: python -m spacy download {model_name}")
    
    
def load_tokenizers(src_lang, trg_lang):
    """
    Loads tokenizers for the specified source and target languages using spaCy.

    This function initializes tokenizers for two given languages, ensuring the required spaCy models 
    are available. If a model is missing, it attempts to install it before loading.

    Args:
        src_lang (str): Source language code (e.g., "en", "fr", "de", "cs", "ar").
        trg_lang (str): Target language code (e.g., "en", "fr", "de", "cs", "ar").

    Returns:
        tuple: A pair of tokenizer functions (src_tokenizer, trg_tokenizer) 
               that tokenize text into a list of lowercase tokens.

    Raises:
        ValueError: If the provided language code is not supported.
        OSError: If spaCy fails to load a required model.
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
        """
        Tokenizes input text using a given spaCy model.

        Args:
            text (str): The input sentence.
            spacy_model (spacy.Language): Loaded spaCy language model.

        Returns:
            list: A list of lowercase tokens.
        """
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
    Builds a vocabulary from a dataset using a given tokenizer.

    This function extracts tokens from a dataset, filters them based on frequency, and constructs 
    a vocabulary with special tokens for padding, sentence start, sentence end, and unknown words.

    Args:
        dataset (iterable): An iterable of sentence pairs (source, target).
        tokenizer (callable): A function that tokenizes input text.
        min_freq (int): Minimum frequency required for a token to be included in the vocabulary.
        max_size (int): Maximum vocabulary size.

    Returns:
        torchtext.vocab.Vocab: The constructed vocabulary object.

    Special Tokens:
        - "<pad>": Padding token
        - "<sos>": Start of sentence
        - "<eos>": End of sentence
        - "<unk>": Unknown token (used as the default index)
    """
    def yield_tokens(data):
        """
        Yields tokens from the dataset using the provided tokenizer.

        Args:
            data (iterable): Dataset containing (source, target) sentence pairs.

        Yields:
            list: Tokenized version of the source and target sentences.
        """
        for (src, trg) in data:
            yield tokenizer(src)
            yield tokenizer(trg)

    vocab = build_vocab_from_iterator(
        yield_tokens(dataset),
        min_freq=min_freq,
        max_tokens=max_size,
        specials=["<pad>", "<sos>", "<eos>", "<unk>"]
    )

    vocab.set_default_index(vocab["<unk>"])  # Assign <unk> as the default index for unknown words

    return vocab


def create_decode_function(vocab):
    """
    Creates a function to decode token indices into readable text.

    This function returns a decoder that converts a sequence of token indices into a human-readable 
    sentence, ignoring special tokens such as "<pad>", "<sos>", "<eos>", and "<unk>".

    Args:
        vocab (torchtext.vocab.Vocab): The vocabulary object.

    Returns:
        callable: A function that converts a list of token indices to a string.

    Example:
        ```
        vocab = build_vocab(...)
        decode_fn = create_decode_function(vocab)
        sentence = decode_fn([2, 5, 12, 8])  # Converts indices to a readable sentence
        ```
    """
    index_to_word = {idx: word for word, idx in vocab.get_stoi().items()}  # Reverse mapping

    def decode(indices):
        """
        Converts a sequence of token indices into a human-readable sentence.

        Args:
            indices (list): A list of token indices.

        Returns:
            str: A reconstructed sentence with special tokens removed.
        """
        return " ".join([index_to_word[idx] for idx in indices if index_to_word[idx] not in ["<pad>", "<sos>", "<eos>", "<unk>"]])

    return decode


def load_dataset(src_lang, trg_lang, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE):
    """
    Loads the IWSLT2016 dataset using torchtext.

    This function retrieves the IWSLT2016 dataset and processes it into training, validation, and test sets.
    It also builds vocabularies and tokenizers for both the source and target languages.

    Args:
        src_lang (str): Source language code (e.g., 'en', 'fr', 'de').
        trg_lang (str): Target language code (e.g., 'en', 'fr', 'de').
        min_freq (int, optional): Minimum frequency a token must have to be included in the vocabulary. Default from config.
        max_size (int, optional): Maximum vocabulary size. Default from config.

    Returns:
        tuple:
            - train_data (list of tuples): Training dataset (source sentence, target sentence).
            - valid_data (list of tuples): Validation dataset.
            - test_data (list of tuples): Test dataset.
            - src_vocab (Vocab): Vocabulary object for the source language.
            - trg_vocab (Vocab): Vocabulary object for the target language.
            - src_tokenizer (callable): Tokenizer function for source language.
            - trg_tokenizer (callable): Tokenizer function for target language.
    """
    train_iter, valid_iter, test_iter = IWSLT2016(language_pair=(src_lang, trg_lang))
    train_data, valid_data, test_data = list(train_iter), list(valid_iter), list(test_iter)

    src_tokenizer, trg_tokenizer = load_tokenizers(src_lang, trg_lang)
    src_vocab = build_vocab(train_data, src_tokenizer, min_freq=min_freq, max_size=max_size)
    trg_vocab = build_vocab(train_data, trg_tokenizer, min_freq=min_freq, max_size=max_size)

    src_tokenizer.decode = create_decode_function(src_vocab)
    trg_tokenizer.decode = create_decode_function(trg_vocab)

    return train_data, valid_data, test_data, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer


def load_dataset_fallback(src_lang, trg_lang, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE, dir=config.DATASET_DIR):
    """
    Loads a locally stored IWSLT2016 dataset.

    This function reads the dataset from local files in case online access is unavailable.
    It supports both plain text and XML-based datasets and processes them accordingly.

    Args:
        src_lang (str): Source language code (e.g., 'en', 'fr', 'de').
        trg_lang (str): Target language code (e.g., 'en', 'fr', 'de').
        min_freq (int, optional): Minimum frequency for vocab tokens. Default from config.
        max_size (int, optional): Maximum vocabulary size. Default from config.
        dir (str, optional): Directory containing the dataset. Default from config.

    Returns:
        tuple:
            - train_data (list of tuples): Training dataset.
            - valid_data (list of tuples): Validation dataset.
            - test_data (list of tuples): Test dataset.
            - src_vocab (Vocab): Source language vocabulary.
            - trg_vocab (Vocab): Target language vocabulary.
            - src_tokenizer (callable): Source language tokenizer function.
            - trg_tokenizer (callable): Target language tokenizer function.

    Raises:
        FileNotFoundError: If the dataset directory or required files are missing.
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
            list of tuples: A list of (source sentence, target sentence) pairs.

        Raises:
            FileNotFoundError: If any of the text dataset files are missing.
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
            list of tuples: A list of (source sentence, target sentence) pairs.

        Raises:
            FileNotFoundError: If the XML dataset files are missing.
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
    
    This function tokenizes source and target sentences, converts them into indices,
    adds start-of-sequence (<sos>) and end-of-sequence (<eos>) tokens, and pads sequences
    to the maximum length within the batch.
    
    Args:
        batch (list of tuples): Each tuple contains (src_sentence, trg_sentence).
        src_vocab (Vocab): Vocabulary object for source language.
        trg_vocab (Vocab): Vocabulary object for target language.
        src_tokenizer (callable): Tokenizer function for source language.
        trg_tokenizer (callable): Tokenizer function for target language.
        pad_idx (int): Index used for padding sequences.

    Returns:
        tuple: 
            - torch.Tensor: Padded source tensor of shape (batch_size, max_src_len).
            - torch.Tensor: Padded target tensor of shape (batch_size, max_trg_len).
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
    Creates PyTorch dataloaders for training, validation, and testing.
    
    The dataloaders use a custom collate function for dynamic sequence padding,
    ensuring that sequences in each batch are padded to the length of the longest
    sequence in that batch.

    Args:
        train_data (list of tuples): Training dataset containing (src_sentence, trg_sentence) pairs.
        valid_data (list of tuples): Validation dataset containing (src_sentence, trg_sentence) pairs.
        test_data (list of tuples): Test dataset containing (src_sentence, trg_sentence) pairs.
        src_vocab (Vocab): Vocabulary object for the source language.
        trg_vocab (Vocab): Vocabulary object for the target language.
        src_tokenizer (callable): Tokenizer function for the source language.
        trg_tokenizer (callable): Tokenizer function for the target language.
        batch_size (int, optional): Batch size for dataloaders. Defaults to config.BATCH_SIZE.

    Returns:
        tuple:
            - DataLoader: DataLoader for the training dataset.
            - DataLoader: DataLoader for the validation dataset.
            - DataLoader: DataLoader for the test dataset.
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