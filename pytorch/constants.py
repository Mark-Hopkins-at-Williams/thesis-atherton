import torch
import evaluate as metrics_import

DATA_DIR = "/home/data/mt_data/nunavut_hansad_Inuktitut–English/Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0/"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRC_LANGUAGE = 'grn'
TGT_LANGUAGE = 'eng'

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

BATCH_SIZE = 32
NUM_EPOCHS = 30

VOCAB_SIZE = 500
ALGORITHM = "BPE_DROPOUT" #last to run
BPE_DROPOUT_RATE = 0.1

METRIC_BLEU = metrics_import.load("sacrebleu")
METRIC_CHRF = metrics_import.load("chrf")