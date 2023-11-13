import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from data import TranslationDataset
from transformers import BertTokenizerFast
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel, BertLMHeadModel
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import evaluate


# print("running test file")
# METRIC_BLEU = evaluate.load("bleu")
# METRIC_CHRF = evaluate.load("chrf")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# en_tokenizer = AutoTokenizer.from_pretrained("tokenizers/en_tok")
# de_tokenizer = AutoTokenizer.from_pretrained("tokenizers/de_tok")

# model = EncoderDecoderModel.from_pretrained("modelsavelocation/")
# print(model)
# VALIDATION_TARGET_WORDS = []
# with open("data/test.fr") as f:
#     for word in f:
#         VALIDATION_TARGET_WORDS.append([word])

# VALIDATION_SOURCE_WORDS = []
# with open("data/test.en") as f:
#     for word in f:
#         VALIDATION_SOURCE_WORDS.append(word)

# def compute_bleu_chrf(target, source, start_tokenizer, end_tokenizer, enc_dec_model):  
#     translations = []
#     for word in source:
#         inputs = start_tokenizer(word, return_tensors="pt").input_ids
#         outputs = enc_dec_model.generate(inputs)#, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95, decoder_start_token_id=6)
#         translation = end_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         translations.append(translation)
#     print("CHRF score: ", METRIC_CHRF.compute(predictions=translations, references=target))
#     print("BLEU score: ", METRIC_BLEU.compute(predictions=translations, references=target))

# compute_bleu_chrf(VALIDATION_TARGET_WORDS, VALIDATION_SOURCE_WORDS, en_tokenizer, de_tokenizer, model)

model_checkpoint = "modelsavelocation"
translator = pipeline("translation", model=model_checkpoint)
print(translator)
translator("they bofrimize us")