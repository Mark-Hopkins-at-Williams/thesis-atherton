import torch
from constants import DEVICE, SRC_LANGUAGE, TGT_LANGUAGE, DATA_DIR
from constants import TOKENIZER_FILE
from transformers import PreTrainedTokenizerFast
from trainutil import sequential_transforms, tensor_transform
from paralleldata import create_hf_dataset, parallel_data_iterator
from trainutil import generate_square_subsequent_mask
from trainutil import sequential_transforms, tensor_transform
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import evaluate

tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{TOKENIZER_FILE}")
token_transform = {}
token_transform[SRC_LANGUAGE] = tokenizer
token_transform[TGT_LANGUAGE] = tokenizer
SRC_VOCAB_SIZE = len(token_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(token_transform[TGT_LANGUAGE])

dataset = create_hf_dataset(DATA_DIR, SRC_LANGUAGE, TGT_LANGUAGE)

transformer = torch.load("checkpoint.pt")
transformer.eval()



# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               lambda x: x['input_ids'], #Numericalization
                                               lambda tok_ids: tensor_transform(tok_ids, tokenizer.bos_token_id, tokenizer.eos_token_id)) # Add BOS/EOS and create tensor

def collate_fn(batch):
    """Collates data samples into batch tensors."""
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=tokenizer.pad_token_id)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tokenizer.pad_token_id)
    return src_batch, tgt_batch

dev_iter = parallel_data_iterator(dataset, SRC_LANGUAGE, TGT_LANGUAGE, split="validation")
dev_dataloader = DataLoader(dev_iter, batch_size=64, collate_fn=collate_fn)


def translate(model: torch.nn.Module, dataloader):
    """Translates source sentence into target language."""
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, src.shape[1]).fill_(start_symbol).type(torch.long).to(DEVICE)
        for i in range(max_len-1):
            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(dim=0)
            ys = torch.cat([ys,
                            next_word], dim=0)
        return ys

    model.eval()
    translations = []
    for src, tgt in dataloader:
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model,  src, src_mask, max_len=num_tokens + 5, start_symbol=tokenizer.bos_token_id)
        for i in range(tgt_tokens.shape[1]):
            translation = token_transform[TGT_LANGUAGE].decode(tgt_tokens[:,i]).replace("<s>", "").replace("</s>", "")
            src_text = token_transform[SRC_LANGUAGE].decode(src[:,i]).replace("<pad>", "").replace("<s>", "").replace("</s>", "")
            ref = token_transform[TGT_LANGUAGE].decode(tgt[:,i]).replace("<pad>", "").replace("<s>", "").replace("</s>", "")
            translations.append({'src': src_text, 'ref': ref, 'hyp': translation})
    return translations

translated = translate(transformer, dev_dataloader)
predictions = [trans['hyp'] for trans in translated]
references = [[trans['ref']] for trans in translated]
sacrebleu = evaluate.load("chrf")
results = sacrebleu.compute(predictions=predictions, references=references)
print(results)

