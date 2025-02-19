# Transformer-From-Scratch

A fully implemented Transformer model built from scratch using PyTorch, designed for neural machine translation (NMT). The model supports bidirectional translation between English and French, German, Czech, and Arabic.

## Features
- Implements the Transformer architecture from scratch.
- Supports translation between English and four other languages: French, German, Czech, and Arabic.
- Enables training from scratch with dataset handling.
- Allows saving and loading trained models.

## Installation

Ensure you have Python 3.8+ installed, then install dependencies:

```sh
pip install torch torchtext tensorboard
```

## Usage

### Training the Model
Run the script and select the language pair for translation:

```sh
python train.py --src en --tgt fr
```

You'll be prompted to select a source and target language from the following options:

```python
valid_langs = {
    "en": ["fr", "de", "cs", "ar"],
    "fr": ["en"],
    "de": ["en"],
    "cs": ["en"],
    "ar": ["en"]
}
```

Once selected, the training process will begin.

### Testing the Model
After training, you can evaluate the model using:

```sh
python test.py --src en --tgt fr --model_path saved_model.pth
```

### Saving & Loading the Model
The model can be saved after training and loaded later for inference:

```sh
python train.py --save_model saved_model.pth
python train.py --load_model saved_model.pth
```

## Dataset

This project uses the IWSLT2016 dataset. If the automated download fails due to a `404 Client Error`, you can manually download it from [wit3.fbk.eu](https://wit3.fbk.eu/2016-01) and extract the required language pairs into the `IWSLT2016/` directory.

## Transformer Architecture Breakdown

The model consists of:
- **Self-Attention Mechanism**: Enables context-aware word representation.
- **Multi-Head Attention**: Improves feature extraction.
- **Positional Encoding**: Provides sequence information.
- **Feedforward Networks**: Enhances expressiveness.
- **Layer Normalization & Dropout**: Prevents overfitting.

## Future Work
- Implement beam search decoding for better translations.
- Optimize model efficiency with quantization and pruning.
- Add support for larger datasets and fine-tuning on domain-specific corpora.

## Contributions
Feel free to fork this repository and submit pull requests if you’d like to contribute!