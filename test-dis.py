import os
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils_squad import read_squad_examples, convert_examples_to_features, RawResult, write_predictions
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm

# 1. Load a trained model
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/BT-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/BT-2-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/EDA-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/ADVANCED-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/senMixup-none-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/senMixup-BT-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/senMixup-none-epo2-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/epo2-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/senMixup-BT-epo2-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/senMixup-none-epo2-2-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/senMixup-none-epo4-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/aug-output/senMixup-EDA-epo2-distilbert-base-uncased_output'
# OUTPUT_DIR = '/home/spl_chae/smbdir/QA/epo4-distilbert-base-uncased_output'
OUTPUT_DIR = '/home/spl_chae/smbdir/QA/Combined_early_epo4_distilbert-base-uncased'

assert os.path.exists(OUTPUT_DIR), "OUTPUT_DIR does not exist."

print("Loading model...")
model = AutoModelForQuestionAnswering.from_pretrained(OUTPUT_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model loaded.")

# 2. Load and pre-process the test set
dev_file = "/home/spl_chae/smbdir/QA/SQuAD/dev-v1.1.json"
assert os.path.exists(dev_file), "Development file does not exist."

predict_batch_size = 32
print("Loading and processing evaluation data...")
eval_examples = read_squad_examples(input_file=dev_file, is_training=False, version_2_with_negative=False)
print(f"Number of evaluation examples: {len(eval_examples)}")

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
eval_features = list(
    tqdm(
        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False
        ),
        desc="Tokenizing examples"
    )
)

print("Preparing TensorDataset...")
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)

eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=predict_batch_size)
print("TensorDataset prepared.")

# 3. Run inference on the test set
print("Running inference...")
model.eval()
all_results = []

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids, input_mask, example_indices = batch
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=input_mask)
        batch_start_logits, batch_end_logits = outputs.start_logits, outputs.end_logits

    for i, example_index in enumerate(example_indices):
        start_logits = batch_start_logits[i].detach().cpu().tolist()
        end_logits = batch_end_logits[i].detach().cpu().tolist()
        eval_feature = eval_features[example_index.item()]
        unique_id = int(eval_feature.unique_id)
        all_results.append(RawResult(
            unique_id=unique_id,
            start_logits=start_logits,
            end_logits=end_logits
        ))

# 4. Write predictions to files
print("Writing predictions...")
output_prediction_file = os.path.join(OUTPUT_DIR, "predictions.json")
output_nbest_file = os.path.join(OUTPUT_DIR, "nbest_predictions.json")
output_null_log_odds_file = os.path.join(OUTPUT_DIR, "null_odds.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
write_predictions(eval_examples, eval_features, all_results, 20, 30, True,
                  output_prediction_file, output_nbest_file, output_null_log_odds_file,
                  True, False, 0.0)

print("Predictions written to", output_prediction_file)
