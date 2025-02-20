import os
import shutil
import torch
from torch.cuda.amp import autocast
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.metrics import bleu_score
import config


def clear_directories(directories=config.DIRECTORIES):
    """
    Deletes all directories specified in the configuration file.
    
    This is useful for clearing previous training outputs, ensuring
    that new experiments start fresh without leftover data.
    """
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"{directory}/ deleted successfully!")


def get_language_choice():
    """
    Prompts the user to select a source and target language from a predefined set of valid languages.
    
    This function performs the following:
    - Asks the user to choose a source language from a set of available options.
    - Dynamically displays the available target languages for the chosen source language.
    - Ensures that the user selects a valid source and target language combination.
    
    Returns:
        tuple: A tuple containing the source language and the target language.
    """
    
    valid_langs = {
        "en": ["fr", "de", "cs", "ar"],
        "fr": ["en"],
        "de": ["en"],
        "cs": ["en"],
        "ar": ["en"]
    }
    
    while True:
        print(f"Please choose a source language from the following options:\n- {'\n- '.join(valid_langs.keys())}")
        src_lang = input("Enter source language: ").strip().lower()
        
        if src_lang in valid_langs:
            break
        print("Invalid source language. Please try again.")
    
    while True:
        print(f"You can translate from {src_lang} to:\n- {'\n- '.join(valid_langs[src_lang])}")
        trg_lang = input("Enter target language: ").strip().lower()
        
        if trg_lang in valid_langs[src_lang]:
            break
        print("Invalid target language. Please try again.")
    
    return src_lang, trg_lang


def get_checkpoint_filename(dir, epoch, src_lang, trg_lang):
    """
    Constructs the checkpoint filename based on the epoch, source language, and target language.

    Args:
        dir (str): The directory where the model checkpoints are stored.
        epoch (int): The epoch number of the model checkpoint.
        src_lang (str): The source language code (e.g., 'en').
        trg_lang (str): The target language code (e.g., 'fr').

    Returns:
        str: The full file path of the checkpoint.
    """
    filename = f"{epoch}_{src_lang}-{trg_lang}_model.pth"
    return os.path.join(dir, filename)


def save_checkpoint(epoch, src_lang, trg_lang, model, optimizer, dir=config.MODELS_DIR):
    """
    Saves the model and optimizer states as a checkpoint.

    Args:
        epoch (int): Epoch number.
        src_lang (str): Source language.
        trg_lang (str): Target language.
        model (torch.nn.Module): The model whose state needs to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state needs to be saved.
        dir (str, optional): Directory to store the checkpoint. Defaults to config.MODELS_DIR.
    """
    print("Saving checkpoint......")
    os.makedirs(dir, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    filepath = get_checkpoint_filename(dir, epoch, src_lang, trg_lang)
    torch.save(checkpoint, filepath)
    print("Checkpoint saved successfully.")


def load_checkpoint(model, optimizer, dir=config.MODELS_DIR):
    """
    Loads a saved model checkpoint.

    Args:
        model (torch.nn.Module): The model where the checkpoint is loaded.
        optimizer (torch.optim.Optimizer): The optimizer where the checkpoint is loaded.
        dir (str, optional): Directory where the checkpoint is stored. Defaults to config.MODELS_DIR.

    Warning:
        If the checkpoint file does not exist, the function prints a warning and does not modify the model.
    """
    src_lang, trg_lang = get_language_choice()

    while True:
        epoch = input("Which epoch would you like to load the model from: ").strip().lower()

        if epoch.isdigit():
            epoch = int(epoch)
            break
        
        print("Invalid input. Please enter again.")
    
    checkpoint_path = get_checkpoint_filename(dir, epoch, src_lang, trg_lang)

    if not os.path.isfile(checkpoint_path):
        print(f"Warning: Checkpoint file '{checkpoint_path}' not found. Falling back without loading checkpoint.")
        return

    print("Loading checkpoint......")
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Checkpoint loaded successfully.")


def tensor_to_tokens(tensor, vocab):
    """
    Converts a tensor of token indices into a list of corresponding tokens using the vocabulary.

    Args:
        tensor (torch.Tensor or list): A tensor or list containing token indices.
        vocab (torchtext.vocab.Vocab): The vocabulary object that maps indices to tokens.

    Returns:
        list: A list of token strings corresponding to the given tensor indices.
    """
    if isinstance(tensor, list):
        return [vocab.lookup_token(idx) for idx in tensor]
    return [vocab.lookup_token(idx) for idx in tensor.tolist()]


def tensor_to_sentence(tensor, vocab):
    """
    Converts a tensor of token indices into a human-readable sentence.

    Args:
        tensor (torch.Tensor or list): A tensor or list containing token indices.
        vocab (torchtext.vocab.Vocab): The vocabulary object that maps indices to tokens.

    Returns:
        str: A sentence string with special tokens ("<sos>", "<eos>", "<pad>", "<unk>") removed.
    """
    tokens = tensor_to_tokens(tensor, vocab)
    tokens = [tok for tok in tokens if tok not in ["<sos>", "<eos>", "<pad>", "<unk>"]]
    return " ".join(tokens)
    

def translate_sentence(model, src, trg_vocab, device, max_length=10):
    """
    Translates a source sentence using a trained model.

    Args:
        model (torch.nn.Module): The trained Transformer model.
        src (torch.Tensor): The source sentence as a tensor of token indices.
        trg_vocab (torchtext.vocab.Vocab): The target vocabulary for mapping indices to words.
        device (torch.device): The device (CPU or GPU) to run inference on.
        max_length (int, optional): The maximum length of the generated translation. Defaults to 10.

    Returns:
        str: The translated sentence as a string.
    """
    src = src.unsqueeze(0).to(device)
    outputs = [trg_vocab["<sos>"]]

    with torch.no_grad():
        for _ in range(max_length):
            trg = torch.LongTensor(outputs).unsqueeze(0).to(device)
            output = model(src, trg)
            best_guess = output[:, -1, :].argmax(dim=-1).item()
            outputs.append(best_guess)

            if best_guess == trg_vocab["<eos>"]:
                break

    translated_sentence = tensor_to_sentence(outputs, trg_vocab)
    return translated_sentence


def calculate_bleu(outputs, targets):
    """
    Computes the BLEU (Bilingual Evaluation Understudy) score to measure translation quality.

    Args:
        outputs (list of list of str): A list of generated translations (tokenized).
        targets (list of list of list of str): A list of reference translations (each reference is a tokenized list).

    Returns:
        float: The BLEU score (0 to 1), where 1 represents perfect translation accuracy.
    """
    return bleu_score(outputs, targets)