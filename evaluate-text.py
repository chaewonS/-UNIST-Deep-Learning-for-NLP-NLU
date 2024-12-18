import json

def load_predictions(file_path):
    """Load predictions from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def get_representative_answers(dataset_file, non_mixup_predictions_file, mixup_predictions_file, num_samples=20):
    """Print representative questions and answers for MixUp and Non-MixUp."""
    # Load dataset and predictions
    with open(dataset_file, "r") as f:
        dataset = json.load(f)['data']
    non_mixup_predictions = load_predictions(non_mixup_predictions_file)
    mixup_predictions = load_predictions(mixup_predictions_file)

    representative_answers = []

    print("\nCollecting representative answers...")

    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                qid = qa['id']
                ground_truths = [ans['text'] for ans in qa['answers']]

                # Get predictions from both Non-MixUp and MixUp
                non_mixup_prediction = non_mixup_predictions.get(qid, "N/A")
                mixup_prediction = mixup_predictions.get(qid, "N/A")

                # Store results
                representative_answers.append({
                    'question': question,
                    'non_mixup_prediction': non_mixup_prediction,
                    'mixup_prediction': mixup_prediction,
                    'ground_truths': ground_truths
                })

                # Stop collecting when we have enough samples
                if len(representative_answers) >= num_samples:
                    break
            if len(representative_answers) >= num_samples:
                break

    # Print formatted results
    for i, item in enumerate(representative_answers, 1):
        print(f"\nQuestion {i}: {item['question']}")
        print(f"- Non-MixUp Prediction: {item['non_mixup_prediction']}")
        print(f"- MixUp Prediction: {item['mixup_prediction']}")
        print(f"- Ground Truths: {item['ground_truths']}")

# Paths to dataset and predictions
dataset_file = "./SQuAD/dev-v1.1.json"  # Replace with the path to your dataset JSON
non_mixup_predictions_file = "./albert-base-v2_output/predictions.json"
mixup_predictions_file = "./aug-output/senMixup-none-albert-base-v2_output/predictions.json"

# Call the function
get_representative_answers(dataset_file, non_mixup_predictions_file, mixup_predictions_file, num_samples=20)
