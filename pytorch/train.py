from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
import os
from model import Seq2SeqTransformer
from constants import DEVICE, SRC_LANGUAGE, TGT_LANGUAGE, DATA_DIR
from constants import EMB_SIZE, NHEAD, FFN_HID_DIM, BATCH_SIZE
from constants import NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NUM_EPOCHS
from constants import ALGORITHM
from constants import METRIC_CHRF, METRIC_BLEU
from paralleldata import train_tokenizer_with_algo
from paralleldata import create_hf_dataset, parallel_data_iterator
from trainutil import generate_square_subsequent_mask, create_mask
from trainutil import sequential_transforms, tensor_transform
from tqdm import tqdm

torch.manual_seed(0)

# Creates the tokenizer for source and target.
print("####Training tokenizer####")
tokenizer = train_tokenizer_with_algo()
token_transform = {}
token_transform[SRC_LANGUAGE] = tokenizer
token_transform[TGT_LANGUAGE] = tokenizer
SRC_VOCAB_SIZE = len(token_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(token_transform[TGT_LANGUAGE])
print("####Tokenizer trained####")

# Loads in the (Huggingface-style) dataset
print("####Loading dataset####")
dataset = create_hf_dataset(DATA_DIR, SRC_LANGUAGE, TGT_LANGUAGE)
print(dataset)
print(dataset["train"][0])
print(dataset["validation"][0])
print(dataset["test"][0])
print("####Dataset loaded####")

# Initializes the transformer model.
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
transformer = transformer.to(DEVICE)

# Initializes the loss function and optimizer.
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
print("####Transforming text####")
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                            lambda x: x['input_ids'], #Numericalization
                                            lambda tok_ids: tensor_transform(tok_ids, tokenizer.bos_token_id, tokenizer.eos_token_id)) # Add BOS/EOS and create tensor
print("####Text transformed####")

def collate_fn(batch):
    """Collates data samples into batch tensors."""
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=tokenizer.pad_token_id)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tokenizer.pad_token_id)
    return src_batch, tgt_batch


def train_epoch(model, optimizer, src_language=SRC_LANGUAGE, tgt_langauge=TGT_LANGUAGE, batch_size=BATCH_SIZE):
    """Trains the model for a single epoch."""
    model.train()
    losses = 0
    train_iter = parallel_data_iterator(dataset, src_language, tgt_langauge, split="train")
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, tokenizer.pad_token_id)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(list(train_dataloader))


def evaluate(model, src_language=SRC_LANGUAGE, tgt_langauge=TGT_LANGUAGE, batch_size=BATCH_SIZE):
    """Evaluates the quality of the current model on the validation set."""
    model.eval()
    losses = 0
    val_iter = parallel_data_iterator(dataset, src_language, tgt_langauge, split="validation")
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)
    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, tokenizer.pad_token_id)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(list(val_dataloader))


def translate(model: torch.nn.Module, src_sentence: str, src_language=SRC_LANGUAGE, tgt_language=TGT_LANGUAGE):
    """Translates source sentence into target language."""
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        for i in range(max_len-1):
            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == tokenizer.eos_token_id:
                break
        return ys

    model.eval()
    src = text_transform[src_language](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=tokenizer.bos_token_id).flatten()
    return token_transform[tgt_language].decode(tgt_tokens).replace("<s>", "").replace("</s>", "")

def evaluate_bleu_chrf(model, dataset, verbose=False, src=SRC_LANGUAGE, tgt=TGT_LANGUAGE):    
    translations = []
    targets = []
    test_dataset = dataset["test"]
    for word_idx in tqdm(range(test_dataset.num_rows)):
        targets.append([test_dataset[word_idx]["translation"][tgt]])
        source = test_dataset[word_idx]["translation"][src]
        translations.append(translate(model, source))
        if verbose and word_idx < 200: print(f"Target: {targets[-1][0]}, Source: {source}, Translation: {translations[-1]}")
    chrf_score = METRIC_CHRF.compute(predictions=translations, references=targets)
    bleu_score = METRIC_BLEU.compute(predictions=translations, references=targets)
    print(f"CHRF score: {chrf_score}")
    print(f"BLEU score: {bleu_score}")
    return chrf_score, bleu_score

def train_model(evalute_metrics = False, verbose = True):
    print("####Starting training loop####")
    #add patience
    val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        if verbose and epoch < 200: print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    if evalute_metrics: evaluate_bleu_chrf(transformer, dataset)
    return NUM_EPOCHS, val_loss

def save_model_cpt(src=SRC_LANGUAGE, tgt=TGT_LANGUAGE, alg=ALGORITHM):
    model_save_location = f"models/{src}_to_{tgt}/{alg}/cpt.pt"

    if not os.path.exists(model_save_location[:-6]):
        os.makedirs(model_save_location[:-6])
    torch.save(transformer, model_save_location)

    loaded_model = torch.load(model_save_location)
    loaded_model.eval()
    evaluate_bleu_chrf(loaded_model, dataset, verbose=True)

train_model()
save_model_cpt()