from textaugment import EDA
import json
from tqdm import tqdm

def augment_eda(input_file, output_file, num_aug=4):
    """
    Simplified EDA-based data augmentation with fallback for empty word lists.
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    eda = EDA()
    augmented_data = {"version": data.get("version", "1.1"), "data": []}

    # Process articles
    for article in tqdm(data["data"], desc="Processing Articles"):
        augmented_article = {"title": article["title"], "paragraphs": []}
        for paragraph in article["paragraphs"]:
            augmented_paragraph = {"context": paragraph["context"], "qas": []}
            for qa in paragraph["qas"]:
                question = qa["question"]
                answers = qa["answers"]

                # Add the original question
                augmented_paragraph["qas"].append(qa)

                # Generate augmented questions
                for _ in range(num_aug):
                    # Apply random EDA techniques
                    augmented_question = question  # Default to original question
                    aug_methods = [
                        lambda q: eda.synonym_replacement(q),
                        lambda q: eda.random_deletion(q, p=0.1),
                        lambda q: eda.random_swap(q),
                        lambda q: eda.random_insertion(q)
                    ]
                    try:
                        augmented_question = aug_methods[_ % len(aug_methods)](question)
                    except ValueError:
                        # Fallback if random insertion or other methods fail
                        pass

                    # Add augmented question to dataset
                    augmented_paragraph["qas"].append({
                        "question": augmented_question,
                        "id": f"{qa['id']}-eda-{_}",
                        "answers": answers
                    })

            augmented_article["paragraphs"].append(augmented_paragraph)
        augmented_data["data"].append(augmented_article)

    # Save augmented dataset
    with open(output_file, "w") as f:
        json.dump(augmented_data, f, indent=2)

    print(f"EDA augmented data saved to {output_file}")

# 실행: EDA 증강 데이터 생성
augment_eda(
    input_file="/home/spl_chae/smbdir/QA/SQuAD/train-v1.1.json",
    output_file="/home/spl_chae/smbdir/QA/SQuAD/eda_4_train-v1.1.json",
    num_aug=4
)
