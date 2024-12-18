# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

# import utils

# def evaluate_london_baseline(eval_corpus_path):
#     with open(eval_corpus_path, encoding='utf-8') as f:
#         data = f.readlines()

#     # Always predict "London"
#     predictions = ["London"] * len(data)

#     # Evaluate accuracy
#     total, correct = utils.evaluate_places(eval_corpus_path, predictions)
#     accuracy = correct / total * 100 if total > 0 else 0.0

#     print(f"Correct: {correct} out of {total}: {accuracy:.2f}%")

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("eval_corpus_path", help="Path to the evaluation corpus.")
#     args = parser.parse_args()

#     evaluate_london_baseline(args.eval_corpus_path)
from tqdm import tqdm
import utils

eval_corpus_path = "birth_dev.tsv"
len_eval = len(open(eval_corpus_path, "r").readlines())
predictions = ["London"] * len_eval

total, correct = utils.evaluate_places(eval_corpus_path, predictions)

if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
else:
    print("No target provided!")