import transformers
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from torch.utils.data import DataLoader
from transformers import pipeline
from transformers import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from huggingface_hub import Repository, get_full_repo_name
from tqdm.auto import tqdm
import torch
from torch import Tensor
import os

"""CONSTANTS"""
MAX_LENGTH = 128
RUN_PROCESS_DATA_TOKENIZER = True
DATASET_PATH = "aatherton2024/eng-nah-svo"
MODEL_CHECKPOINT = "eng-nah-svo-translation"
PRETRAINED_MODEL = "t5-small"
METRIC_BLEU = evaluate.load("sacrebleu")
METRIC_CHRF = evaluate.load("chrf")
ARGS = Seq2SeqTrainingArguments(
    MODEL_CHECKPOINT,
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=True,
    use_cpu=False,
    no_cuda=False
)

"""Simple method to either load tokenizer or train new one"""
def get_tokenizer():
    if not RUN_PROCESS_DATA_TOKENIZER: 
        return AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    def get_training_corpus(raw_datasets):
        return (
            raw_datasets["train"][ds][i : i + 1000]
            for i in range(0, len(raw_datasets["train"]), 1000)
            for ds in ["en", "fr"]
        )

    training_corpus = get_training_corpus(raw_datasets)
    old_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'eos_token': "</s>"})
    tokenizer.save_pretrained(MODEL_CHECKPOINT)
    tokenizer.push_to_hub(MODEL_CHECKPOINT)
    return tokenizer

"""Scan dataset, storing lists of english and french words then returning the tokenization of them"""
def preprocess_function(examples):
    prefix = "translate en to fr"
    inputs = [prefix + example for example in examples["en"]]
    targets = [prefix + example for example in examples["fr"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True, padding=True
    )
    return model_inputs

"""Apply preprocessing in one go to all splits of the dataset"""
def tokenize_datasets():
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    return tokenized_datasets

"""Simple method to return test metrics"""
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result_bleu = METRIC_BLEU.compute(predictions=decoded_preds, references=decoded_labels)
    result_chrf = METRIC_CHRF.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result_bleu["score"], "chrf": result_chrf["score"]}

"""Simple method to evaluate, train, then reevaluate model"""
def evaluate_train_evaluate():
    print(trainer.evaluate(max_length=MAX_LENGTH))
    trainer.train()
    print(trainer.evaluate(max_length=MAX_LENGTH))
    #trainer.push_to_hub(tags="translation", commit_message="Training complete")

"""Method to test translation capabilities of model"""
def test_translation():
    sequence = "he bofrimizes us"
    inputs = tokenizer(sequence, return_tensors="pt").input_ids.to("cuda")
    print(inputs)
    print(tokenizer.tokenize(sequence))
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    print(outputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    translator = pipeline("translation_en_to_fr", model="eng-nah-svo-translation")
    print(translator(sequence))

"""Main testing script"""
raw_datasets = load_dataset("aatherton2024/eng-nah-svo")
tokenizer = get_tokenizer()
tokenized_datasets = tokenize_datasets()
model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    ARGS,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

evaluate_train_evaluate()

test_translation()





























# MODEL_CHECKPOINT = "aatherton2024/eng-nah-svo-translation"
# translator = pipeline("translation", model=MODEL_CHECKPOINT)
# translator("Default to expanded threads")
# print(translator(
#     "you did not frichopize him"
# ))



# tokenized_datasets.set_format("torch")
# train_dataloader = DataLoader(
#     tokenized_datasets["train"],
#     shuffle=True,
#     collate_fn=data_collator,
#     batch_size=8,
# )
# eval_dataloader = DataLoader(
#     tokenized_datasets["test"], collate_fn=data_collator, batch_size=8, drop_last=True
# )

# model = AutoModelForSeq2SeqLM.from_pretrained("eng-nah-svo-translation")


# optimizer = AdamW(model.parameters(), lr=2e-5)



# accelerator = Accelerator()
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )



# num_train_epochs = 3
# num_update_steps_per_epoch = len(train_dataloader)
# num_training_steps = num_train_epochs * num_update_steps_per_epoch

# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )



# model_name = "model"

# output_dir = "./output"
# repo = Repository("/mnt/storage/aatherton/hf_eng_fra_trans", clone_from="aatherton2024/hf_eng_fra_trans")


# def postprocess(predictions, labels):
#     predictions = predictions.cpu().numpy()
#     labels = labels.cpu().numpy()

#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Some simple post-processing
#     decoded_preds = [pred.strip() for pred in decoded_preds]
#     decoded_labels = [[label.strip()] for label in decoded_labels]
#     return decoded_preds, decoded_labels



# progress_bar = tqdm(range(num_training_steps))

# for epoch in range(num_train_epochs):
#     # Training
#     model.train()
#     for batch in train_dataloader:
#         outputs = model(**batch)
#         loss = outputs.loss
#         accelerator.backward(loss)

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)

#     # Evaluation
#     model.eval()
#     for batch in tqdm(eval_dataloader):
#         with torch.no_grad():
#             generated_tokens = accelerator.unwrap_model(model).generate(
#                 batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 MAX_LENGTH=128,
#             )
#         labels = batch["labels"]

#         # Necessary to pad predictions and labels for being gathered
#         generated_tokens = accelerator.pad_across_processes(
#             generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
#         )
#         labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

#         predictions_gathered = accelerator.gather(generated_tokens)
#         labels_gathered = accelerator.gather(labels)

#         decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
#         METRIC_BLEU.add_batch(predictions=decoded_preds, references=decoded_labels)

#     results = METRIC_BLEU.compute()
#     print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

#     # Save and upload
#     accelerator.wait_for_everyone()
#     unwrapped_model = accelerator.unwrap_model(model)
#     unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
#     if accelerator.is_main_process:
#         tokenizer.save_pretrained(output_dir)
#         repo.push_to_hub(
#             commit_message=f"Training in progress epoch {epoch}", blocking=False
#         )



# Replace this with your own checkpoint
