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