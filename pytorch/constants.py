import torch

DATA_DIR = "/home/data/mt_data/tpo/eng-nah-svo/data/"

BPE_TOKENIZER_FILE = "tokenizers/BPE"
UNIGRAM_TOKENIZER_FILE = "tokenizers/UNIGRAM"
WORDPIECE_TOKENIZER_FILE = "tokenizers/WORDPIECE"

BPE_MODEL_FILE = "models/BPE/checkpoint.pt"
UNIGRAM_MODEL_FILE = "models/UNIGRAM/checkpoint.pt"
WORDPIECE_MODEL_FILE = "models/WORDPIECE/checkpoint.pt"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

BATCH_SIZE = 128
NUM_EPOCHS = 40

VOCAB_SIZE = 500
ALGORITHM = "WORDPIECE"