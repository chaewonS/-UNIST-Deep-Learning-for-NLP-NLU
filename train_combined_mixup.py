import os
import random
import time
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from tqdm import tqdm
import numpy as np
import json
from textaugment import EDA  # Make sure to install the textaugment package

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
num_train_epochs = 5  # Maximum epochs
train_batch_size = 4  # Batch size
learning_rate = 3e-5
patience = 2  # Early stopping patience
SQUAD_DIR = '/home/spl_chae/smbdir/QA/SQuAD'
model_names = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "albert-base-v2"]

train_file = os.path.join(SQUAD_DIR, 'train-v1.1.json')
assert os.path.exists(train_file), f"Training file not found at {train_file}"

# Load training data
with open(train_file, "r") as f:
    train_data = json.load(f)["data"]

# Augmentation Functions
def back_translate(data, tgt_lang="de", batch_size=32):
    from transformers import MarianMTModel, MarianTokenizer

    print("Loading translation models...")
    model_name = f"Helsinki-NLP/opus-mt-en-{tgt_lang}"
    back_model_name = f"Helsinki-NLP/opus-mt-{tgt_lang}-en"
    src_tokenizer = MarianTokenizer.from_pretrained(model_name)
    src_model = MarianMTModel.from_pretrained(model_name).to(device)
    tgt_tokenizer = MarianTokenizer.from_pretrained(back_model_name)
    tgt_model = MarianMTModel.from_pretrained(back_model_name).to(device)

    def translate_batch(texts, tokenizer, model):
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        translations = model.generate(**tokens, max_new_tokens=512)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translations]

    augmented_data = []
    for article in tqdm(data, desc="Back-Translation"):
        for paragraph in article["paragraphs"]:
            questions = [qa["question"] for qa in paragraph["qas"]]
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                try:
                    translated = translate_batch(batch_questions, src_tokenizer, src_model)
                    back_translated = translate_batch(translated, tgt_tokenizer, tgt_model)
                    for qa, bt_q in zip(paragraph["qas"][i:i + len(batch_questions)], back_translated):
                        augmented_data.append({
                            "question": bt_q,
                            "context": paragraph["context"],
                            "answers": qa["answers"],
                            "id": qa["id"] + "-bt"
                        })
                except Exception as e:
                    print(f"Error during back-translation: {e}")
    return augmented_data

def augment_eda(data, num_aug=2):
    eda = EDA()
    augmented_data = []
    for article in tqdm(data, desc="EDA"):
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                for _ in range(num_aug):
                    try:
                        aug_question = eda.synonym_replacement(qa["question"])
                        augmented_data.append({
                            "question": aug_question,
                            "context": paragraph["context"],
                            "answers": qa["answers"],
                            "id": qa["id"] + f"-eda-{_}"
                        })
                    except Exception as e:
                        print(f"Error during EDA: {e}")
    return augmented_data

# Perform augmentations
print("Applying Back-Translation...")
bt_augmented = back_translate(random.sample(train_data, len(train_data) // 5))
print(f"Back-Translation completed: {len(bt_augmented)} examples.")

print("Applying EDA...")
eda_augmented = augment_eda(train_data, num_aug=1)
print(f"EDA completed: {len(eda_augmented)} examples.")

# Combine augmented data
augmented_data = bt_augmented + eda_augmented

# Tokenization and Dataset Creation
def process_example(example, model_name, max_seq_length=384, doc_stride=128, max_query_length=64):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(
        example["question"],
        example["context"],
        max_length=max_seq_length,
        truncation=True,
        stride=doc_stride,
        return_tensors="pt",
        padding="max_length",
    )

# Mixup Function
def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Update training loop
for model_name in model_names:
    print(f"\nTraining and saving model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("Tokenizing data...")
    tokenized = [process_example(example, model_name) for example in tqdm(augmented_data)]

    # Prepare dataset
    input_ids = torch.cat([t["input_ids"] for t in tokenized])
    attention_mask = torch.cat([t["attention_mask"] for t in tokenized])
    start_positions = torch.tensor([e["answers"][0]["answer_start"] for e in augmented_data])
    end_positions = start_positions + torch.tensor([len(e["answers"][0]["text"]) for e in augmented_data])

    if "distilbert" in model_name or "roberta" in model_name:
        dataset = TensorDataset(input_ids, attention_mask, start_positions, end_positions)
    else:
        token_type_ids = torch.cat([t["token_type_ids"] for t in tokenized])
        dataset = TensorDataset(input_ids, attention_mask, token_type_ids, start_positions, end_positions)

    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=train_batch_size)

    print("Starting training...")
    model.train()
    best_loss = float("inf")
    patience_counter = 0

    gradient_accumulation_steps = 4

    for epoch in range(num_train_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_train_epochs}")

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)

            if "distilbert" in model_name or "roberta" in model_name:
                inputs, labels_start, labels_end = batch[0], batch[2], batch[3]
            else:
                inputs, labels_start, labels_end = batch[0], batch[3], batch[4]

            # Mixup
            mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels_start, alpha=0.4)
            mixed_inputs = mixed_inputs.long()

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=mixed_inputs, attention_mask=batch[1], start_positions=y_a, end_positions=y_b)
                loss_start = mixup_criterion(criterion, outputs.start_logits, y_a, y_b, lam)
                loss_end = mixup_criterion(criterion, outputs.end_logits, y_a, y_b, lam)
                loss = (loss_start + loss_end) / 2 / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            epoch_iterator.set_description(f"Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} completed. Total Loss: {total_loss:.4f}")
        torch.cuda.empty_cache()
        gc.collect()

        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save model
    output_dir = f"./Combined_early_mixup_epo4_{model_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved at {output_dir}.")