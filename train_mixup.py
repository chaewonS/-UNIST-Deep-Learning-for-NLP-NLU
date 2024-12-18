import time
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from utils_squad import read_squad_examples, convert_examples_to_features
from tqdm import tqdm
from multiprocessing import Pool
import random
import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Training parameters
num_train_epochs = 2
train_batch_size = 16
learning_rate = 3e-5
SQUAD_DIR = '/home/spl_chae/smbdir/QA/SQuAD'
# model_names = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "albert-base-v2"]
model_names = ["albert-base-v2"]
train_file = os.path.join(SQUAD_DIR, 'bt_4_train-v1.1.json')

print("Loading training data...")
train_examples = read_squad_examples(train_file, is_training=True, version_2_with_negative=False)
print(f"Loaded {len(train_examples)} training examples.")

def process_example_serializable(args):
    example, model_name, max_seq_length, doc_stride, max_query_length = args
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return convert_examples_to_features(
        [example], tokenizer,
        max_seq_length=max_seq_length, doc_stride=doc_stride,
        max_query_length=max_query_length, is_training=True
    )[0]

# Apply senMixup during training
def apply_senMixup(logits, alpha=0.4):
    """
    Apply Mixup to logits with the given alpha parameter.
    Args:
        logits: Tensor of model outputs (batch_size, num_classes)
        alpha: Mixup interpolation strength
    Returns:
        Mixed logits
    """
    batch_size = logits.size(0)
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size).to(logits.device)
    mixed_logits = lam * logits + (1 - lam) * logits[index]
    return mixed_logits, lam, index

# Train and save models with senMixup
for model_name in model_names:
    print(f"\nTraining and saving model: {model_name}")
    output_dir = f"/home/spl_chae/smbdir/QA/aug-output/senMixup-BT-epo2-{model_name.replace('/', '_')}_output"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Tokenizing training data using multiprocessing...")
    start_time = time.time()
    with Pool(processes=4) as pool:
        train_features = list(tqdm(
            pool.imap(process_example_serializable, [
                (example, model_name, 384, 128, 64) for example in train_examples
            ]),
            total=len(train_examples),
            desc="Tokenizing"
        ))
    print(f"Tokenizing completed in {time.time() - start_time:.2f} seconds.")

    print("Preparing TensorDataset...")
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    model.train()
    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_train_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_train_epochs}")

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch

            # Model outputs
            if "distilbert" in model_name or "roberta" in model_name:
                outputs = model(input_ids=input_ids, attention_mask=input_mask, start_positions=start_positions, end_positions=end_positions)
            else:
                outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, start_positions=start_positions, end_positions=end_positions)

            # Extract start_logits and end_logits
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Apply senMixup
            start_logits, lam, index = apply_senMixup(start_logits)
            end_logits, _, _ = apply_senMixup(end_logits)

            loss_fn = torch.nn.CrossEntropyLoss()
            loss_start = lam * loss_fn(start_logits, start_positions) + (1 - lam) * loss_fn(start_logits[index], start_positions[index])
            loss_end = lam * loss_fn(end_logits, end_positions) + (1 - lam) * loss_fn(end_logits[index], end_positions[index])

            loss = (loss_start + loss_end) / 2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            epoch_iterator.set_description(f"Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1}/{num_train_epochs} completed. Loss: {total_loss:.4f}")

    print(f"Training for {model_name} completed in {time.time() - start_time:.2f} seconds.")
    print(f"Saving the model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved for {model_name}.")

