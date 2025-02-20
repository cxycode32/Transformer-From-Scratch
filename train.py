import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torchtext
torchtext.disable_torchtext_deprecation_warning()

import config
from data import load_dataset, load_dataset_fallback, create_dataloaders
from model import Transformer
from utils import (
    get_language_choice,
    clear_directories,
    save_checkpoint,
    load_checkpoint,
    tensor_to_sentence,
    translate_sentence,
    calculate_bleu,
)


torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


def train_model():
    clear_directories()
    
    src_lang, trg_lang = get_language_choice()
    print(f"Training model for {src_lang} to {trg_lang} translation...")
    
    """
    This part here, sometime torchtext is unable to load dataset due to the following error:
    
    404 Client Error: Not Found for url: https://drive.usercontent.google.com/download?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8
    This exception is thrown by __iter__ of GDriveReaderDataPipe(skip_on_error=False, source_datapipe=OnDiskCacheHolderIterDataPipe, timeout=None).

    Refer to: https://github.com/pytorch/text/issues/1676
    
    In this case, you will need to download the datasets manually via https://wit3.fbk.eu/2016-01
    After that:
    1. Go to the downloaded folder 2016-01/texts/
    2. You will see a list of source language folders
    3. Each source language folder has its target language folder inside
    4. <src_lang>/<trg_lang>/
    5. Then you will see <src_lang>-<trg_lang>.zip
    6. For example, my src_lang is 'fr' and trg_lang is 'en', then fr/en/fr-en.zip
    7. Extract the fr-en/ folder from the ZIP file to <project_root_dir>/Seq2Seq_Attention/IWSLT2016
    """
    try:
        train_data, valid_data, test_data, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer = load_dataset(src_lang, trg_lang)
    except Exception as e:
        print(f"Warning: Failed to load dataset using torchtext due to {e}. Falling back to local dataset.")
        train_data, valid_data, test_data, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer = load_dataset_fallback(src_lang, trg_lang)

    train_loader, valid_loader, test_loader = create_dataloaders(train_data, valid_data, test_data, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer)    

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)
    
    src_pad_idx = src_vocab["<pad>"]
    trg_pad_idx = trg_vocab["<pad>"]
    
    model = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        config.EMBEDDING_SIZE,
        config.NUM_LAYERS,
        config.NUM_HEADS,
        config.FORWARD_EXPANSION,
        config.DROPOUT,
        config.MAX_LENGTH,
        config.DEVICE,
    ).to(config.DEVICE)
    
    # We are using AdamW here for better weight decay handling
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10
    )
    writer = SummaryWriter(f"{config.LOG_DIR}")
    
    if config.LOAD_MODEL:
        load_checkpoint(model, optimizer)

    step = 0

    for epoch in range(config.EPOCHS_NUM):
        model.train()
        epoch_loss = 0
        
        print(f"EPOCH [{epoch+1}/{config.EPOCHS_NUM}]")

        for idx, (src, trg) in enumerate(train_loader):
            if idx % 100 == 0:
                torch.cuda.empty_cache()

            src, trg = src.to(config.DEVICE, non_blocking=True), trg.to(config.DEVICE, non_blocking=True)
            src, trg = src.clamp(0, src_vocab_size - 1), trg.clamp(0, trg_vocab_size - 1)

            with autocast():
                output = model(src, trg[:, :-1])  # output.shape: (batch, seq_len-1, vocab_size)
                output = output.contiguous().view(-1, output.shape[2])  # (batch * seq_len-1, vocab_size)
                trg = trg[:, 1:].contiguous().view(-1)  # (batch * seq_len-1,)
                optimizer.zero_grad()
                loss = criterion(output, trg)
                epoch_loss += loss.item()
                print(f"[TRAIN LOADER] [{idx+1}/{len(train_loader)}] Loss: {loss.item()}")

            # loss.backward()
            scaler.scale(loss).backward()
            
            # Clip to prevent the exploding gradient issue
            # Exploding Gradients: The issue is caused by gradients growing exponentially as they are backpropagated
            # Weights get too large, leading to unstable training
            # The issue will cause the loss to become NaN/infinity and the training becomes meaningless
            # The following code limits the magnitude of gradients andd scales down gradients if they exceed threshold (max_norm=1) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1
            
        mean_loss = epoch_loss / len(train_loader)
        mean_valid_loss = validate_model(model, valid_loader, src_vocab, trg_vocab, criterion, config.DEVICE)

        writer.add_scalar("Validation loss", mean_valid_loss, global_step=epoch)

        print(f"Epoch {epoch+1}/{config.EPOCHS_NUM}, Loss: {mean_loss:.4f}, Validation Loss: {mean_valid_loss:.4f}")

        if config.SAVE_MODEL:
            save_checkpoint(epoch+1, src_lang, trg_lang, model, optimizer)
            
        scheduler.step(mean_loss)
        
    test_model(model, test_loader, src_vocab, trg_vocab, config.DEVICE)
        
    
def validate_model(model, valid_loader, src_vocab, trg_vocab, criterion, device):
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for idx, (src, trg) in enumerate(valid_loader):
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            src, trg = src.clamp(0, len(src_vocab) - 1), trg.clamp(0, len(trg_vocab) - 1)

            output = model(src, trg[:, :-1])
            output = output.contiguous().view(-1, output.shape[2])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            valid_loss += loss.item()
            print(f"[VALID LOADER] [{idx+1}/{len(valid_loader)}] Valid Loss: {loss.item()}")

    return valid_loss / len(valid_loader)


def test_model(model, test_loader, src_vocab, trg_vocab, device):
    model.eval()
    targets, outputs = [], []
    
    with torch.no_grad():
        for idx, (src, trg) in enumerate(test_loader):
            if idx % 100 == 0:
                torch.cuda.empty_cache()

            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            src, trg = src.clamp(0, len(src_vocab) - 1), trg.clamp(0, len(trg_vocab) - 1)

            batch_size = src.shape[0]

            for i in range(batch_size):
                src_sentence = tensor_to_sentence(src[i], src_vocab)
                trg_sentence = tensor_to_sentence(trg[i], trg_vocab)

                prediction = translate_sentence(model, src[i], trg_vocab, device)

                print(f"-----[{idx+1}/{len(test_loader)}] ({i+1}/{batch_size})----------")
                print(f"SRC: {src_sentence}")
                print(f"TRG: {trg_sentence}")
                print(f"PRED: {prediction}")

                targets.append([trg_sentence])
                outputs.append(prediction)
                
        bleu_score = calculate_bleu(outputs, targets) * 100
        print(f"BLEU Score: {bleu_score:.4f}")
        
        
if __name__ == "__main__":
    train_model()