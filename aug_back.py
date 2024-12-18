from transformers import MarianMTModel, MarianTokenizer
import json
import torch
import random
from tqdm import tqdm

def back_translate(input_file, output_file, src_lang="en", tgt_lang="de", batch_size=32, max_new_tokens=40, sample_rate=0.8):
    """Simplified Back-Translation for QA tasks."""
    print("Initializing models and tokenizers...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load translation models and tokenizers
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    back_model_name = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"

    src_tokenizer = MarianTokenizer.from_pretrained(model_name)
    src_model = MarianMTModel.from_pretrained(model_name).to(device)
    tgt_tokenizer = MarianTokenizer.from_pretrained(back_model_name)
    tgt_model = MarianMTModel.from_pretrained(back_model_name).to(device)

    print("Models and tokenizers are ready.")

    def translate_batch(texts, tokenizer, model):
        """Translate a batch of texts."""
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        translations = model.generate(**tokens, max_new_tokens=max_new_tokens)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translations]

    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"Sampling {int(len(data['data']) * sample_rate)} articles out of {len(data['data'])}...")
    sampled_articles = random.sample(data["data"], int(len(data["data"]) * sample_rate))
    data["data"] = sampled_articles

    augmented_data = {"data": []}
    total_questions = sum(len(paragraph["qas"]) for article in data["data"] for paragraph in article["paragraphs"])
    print(f"Total questions to process: {total_questions}")

    question_progress = tqdm(total=total_questions, desc="Processing Questions")

    for article in data["data"]:
        augmented_article = {"title": article["title"], "paragraphs": []}
        for paragraph in article["paragraphs"]:
            augmented_paragraph = {"context": paragraph["context"], "qas": []}
            
            questions = [qa["question"] for qa in paragraph["qas"]]
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]

                # Translate to target language and back
                translated_batch = translate_batch(batch_questions, src_tokenizer, src_model)
                back_translated_batch = translate_batch(translated_batch, tgt_tokenizer, tgt_model)

                for qa, back_translated in zip(paragraph["qas"][i:i + batch_size], back_translated_batch):
                    augmented_paragraph["qas"].append({
                        "question": back_translated,
                        "id": qa["id"] + "-bt",
                        "answers": qa["answers"]
                    })
                    question_progress.update(1)

            # Include original questions
            augmented_paragraph["qas"].extend(paragraph["qas"])
            augmented_article["paragraphs"].append(augmented_paragraph)
        augmented_data["data"].append(augmented_article)

    question_progress.close()
    print("Translation complete. Saving augmented data...")

    with open(output_file, "w") as f:
        json.dump(augmented_data, f, indent=2)
    print(f"Augmented data saved to {output_file}")

# 실행: Back-Translation 증강 데이터 생성
back_translate(
    input_file="/home/spl_chae/smbdir/QA/SQuAD/train-v1.1.json",
    output_file="/home/spl_chae/smbdir/QA/SQuAD/bt_3_train-v1.1.json",
    tgt_lang="de",  # Single target language (German)
    sample_rate=0.8,  # Use 80% of the data for augmentation
    batch_size=32,  # Larger batch size for efficiency
    max_new_tokens=40  # Keep generated token length reasonable
)
