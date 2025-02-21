# Transformer-From-Scratch

A fully implemented Transformer model built from scratch using PyTorch, designed for neural machine translation (NMT). This project supports translation between English and French, German, Czech, and Arabic, leveraging the IWSLT2016 dataset.


## Features

- **Transformer Architecture from Scratch:** Implements the Transformer model from scratch.
- **IWSLT2016 Dataset:** Uses the IWSLT2016 dataset.
- **Multi-language Translation:** Trains the model for translation from English to French, German, Czech, and Arabic, and vice versa.
- **Training, Validation, and Testing:** Complete training, validation, and testing datasets.
- **Save and Load:** Supports loading and saving trained models.
- **TensorBoard Integration:** Visualize training process with TensorBoard.


## Installation

### Clone the Repository

```bash
git clone https://github.com/cxycode32/Transformer-From-Scratch.git
cd Transformer-From-Scratch/
```

Install dependencies with:
```bash
pip install -r requirements.txt
```


## Usage

### Training the Model

You can go to `config.py` to configure your hyperparameters.

Then run the command:
```bash
python train.py
```

You will be prompted to select a source and target language from the following options:
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

### Saving & Loading the Model
The model can be saved after training and loaded later:

`config.py`:
```python
SAVE_MODEL = True
LOAD_MODEL = True
```


## Dataset

This project uses the IWSLT2016 dataset. If the automated download fails due to a `404 Client Error`, you can manually download it from [wit3.fbk.eu](https://wit3.fbk.eu/2016-01). After that:
1. Go to the downloaded folder 2016-01/texts/
2. You will see a list of source language folders
3. Each source language folder has its target language folder inside
4. <src_lang>/<trg_lang>/
5. Then you will see <src_lang>-<trg_lang>.zip
6. For example, my src_lang is 'fr' and trg_lang is 'en', then fr/en/fr-en.zip
7. Extract the fr-en/ folder from the ZIP file to Transformer-From-Scratch/IWSLT2016


## Credits:

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Repo: [aladdinpersson/Machine-Learning-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection)


## License

This project is open-source under the MIT License.


## How to Contribute

Contributions are welcome! If you'd like to improve the project, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.