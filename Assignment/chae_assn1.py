import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
import nltk
from nltk import pos_tag
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# ----------------------------------------------------------------------------------- #
# 1.1 Distributional Counting

def load_vocab(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def load_word_similarity(file_path):
    word_pairs = []
    similarities = []
    with open(file_path, 'r') as f:
        for line in f.readlines()[1:]:
            word1, word2, similarity = line.strip().split('\t')
            word_pairs.append((word1, word2))
            similarities.append(float(similarity))
    return word_pairs, similarities

def build_word_context_matrix(corpus, vocab, context_vocab, window_size=3):
    C = np.zeros((len(vocab), len(context_vocab)))
    vocab_index = {word: i for i, word in enumerate(vocab)}
    context_vocab_index = {word: i for i, word in enumerate(context_vocab)}
    
    for sentence in corpus:
        words = ['<s>'] * window_size + sentence.split() + ['</s>'] * window_size
        for i, word in enumerate(words[window_size:-window_size], start=window_size):
            if word in vocab_index:
                target_idx = vocab_index[word]
                context_indices = list(range(i - window_size, i)) + list(range(i + 1, i + window_size + 1))
                for j in context_indices:
                    context_word = words[j]
                    if context_word in context_vocab_index:
                        C[target_idx, context_vocab_index[context_word]] += 1
    return C

def evaluate_word_vectors(matrix, word_pairs, human_similarities, vocab):
    cosine_similarities = []
    for word1, word2 in word_pairs:
        if word1 in vocab and word2 in vocab:
            idx1, idx2 = vocab.index(word1), vocab.index(word2)
            vec1, vec2 = matrix[idx1], matrix[idx2]
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0: 
                cosine_similarity = 0.0
            else:
                cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            cosine_similarity = 0.0
        
        cosine_similarities.append(cosine_similarity)
    
    if len(cosine_similarities) > 0:
        spearman_corr, _ = spearmanr(cosine_similarities, human_similarities)
    else:
        spearman_corr = 0.0 
    
    return spearman_corr

def run_distributional_counting():
    vocab_wordsim = load_vocab('./assn1/vocab-wordsim.txt')
    vocab_25k = load_vocab('./assn1/vocab-25k.txt')
    corpus = load_vocab('./wiki-1percent.txt/wiki-1percent.txt')
    
    C = build_word_context_matrix(corpus, vocab_wordsim, vocab_25k)
    
    men_pairs, men_similarities = load_word_similarity('./assn1/men.txt')
    simlex_pairs, simlex_similarities = load_word_similarity('./assn1/simlex-999.txt')
    
    men_spearman_C = evaluate_word_vectors(C, men_pairs, men_similarities, vocab_wordsim)
    simlex_spearman_C = evaluate_word_vectors(C, simlex_pairs, simlex_similarities, vocab_wordsim)
    
    print(f"Spearman Correlation for MEN (C): {men_spearman_C}")
    print(f"Spearman Correlation for SimLex-999 (C): {simlex_spearman_C}")

# ----------------------------------------------------------------------------------- #
# 1.2 Computing PMIs

def compute_pmi(C):
    total_count = np.sum(C)
    word_counts = np.sum(C, axis=1)
    context_counts = np.sum(C, axis=0)
    Cpmi = np.zeros_like(C)
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] > 0:
                p_wc = C[i, j] / total_count
                p_w = word_counts[i] / total_count
                p_c = context_counts[j] / total_count
                Cpmi[i, j] = max(np.log(p_wc / (p_w * p_c)), 0)
    
    return Cpmi

def run_pmi():
    vocab_wordsim = load_vocab('./assn1/vocab-wordsim.txt')
    vocab_25k = load_vocab('./assn1/vocab-25k.txt')
    corpus = load_vocab('./wiki-1percent.txt/wiki-1percent.txt')
    
    C = build_word_context_matrix(corpus, vocab_wordsim, vocab_25k)
    Cpmi = compute_pmi(C)
    
    men_pairs, men_similarities = load_word_similarity('./assn1/men.txt')
    simlex_pairs, simlex_similarities = load_word_similarity('./assn1/simlex-999.txt')
    
    men_spearman_Cpmi = evaluate_word_vectors(Cpmi, men_pairs, men_similarities, vocab_wordsim)
    simlex_spearman_Cpmi = evaluate_word_vectors(Cpmi, simlex_pairs, simlex_similarities, vocab_wordsim)
    
    print(f"Spearman Correlation for MEN (Cpmi): {men_spearman_Cpmi}")
    print(f"Spearman Correlation for SimLex-999 (Cpmi): {simlex_spearman_Cpmi}")

# ----------------------------------------------------------------------------------- #
# 1.3 Experimentation

def run_experimentation():
    vocab_wordsim = load_vocab('./assn1/vocab-wordsim.txt')
    vocab_25k = load_vocab('./assn1/vocab-25k.txt')
    corpus = load_vocab('./wiki-1percent.txt/wiki-1percent.txt')
    
    men_pairs, men_similarities = load_word_similarity('./assn1/men.txt')
    simlex_pairs, simlex_similarities = load_word_similarity('./assn1/simlex-999.txt')

    for w in [1, 3, 6]:
        C = build_word_context_matrix(corpus, vocab_wordsim, vocab_25k, window_size=w)
        Cpmi = compute_pmi(C)
        
        men_spearman_C = evaluate_word_vectors(C, men_pairs, men_similarities, vocab_wordsim)
        simlex_spearman_C = evaluate_word_vectors(C, simlex_pairs, simlex_similarities, vocab_wordsim)
        
        print(f"Window Size {w} - MEN (C): {men_spearman_C}")
        print(f"Window Size {w} - SimLex-999 (C): {simlex_spearman_C}")
        
        men_spearman_Cpmi = evaluate_word_vectors(Cpmi, men_pairs, men_similarities, vocab_wordsim)
        simlex_spearman_Cpmi = evaluate_word_vectors(Cpmi, simlex_pairs, simlex_similarities, vocab_wordsim)
        
        print(f"Window Size {w} - MEN (Cpmi): {men_spearman_Cpmi}")
        print(f"Window Size {w} - SimLex-999 (Cpmi): {simlex_spearman_Cpmi}")

# ----------------------------------------------------------------------------------- #
# 1.4 Analysis
# 1.4.1 Warm-up: Printing nearest neighbors

def print_nearest_neighbors(word, word_vectors, vocab, top_n=10):
    if word not in vocab:
        print(f"{word} not found in vocabulary.")
        return

    word_idx = vocab.index(word)
    similarities = cosine_similarity([word_vectors[word_idx]], word_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1][1:top_n+1]
    
    neighbors = [vocab[i] for i in sorted_indices if len(vocab[i]) > 2 and vocab[i].isalpha()] 
    print(f"Nearest neighbors for '{word}': {neighbors[:top_n]}")

def run_nearest_neighbors():
    vocab_25k = load_vocab('./assn1/vocab-25k.txt')
    corpus = load_vocab('./wiki-1percent.txt/wiki-1percent.txt')
    
    for w in [1, 6]:
        Cpmi = build_word_context_matrix(corpus, vocab_25k, vocab_25k, window_size=w)
        Cpmi = compute_pmi(Cpmi)
        print(f"Window Size {w}")
        print_nearest_neighbors('monster', Cpmi, vocab_25k)

# ----------------------------------------------------------------------------------- #
# 1.4.2 Part-of-speech tag similarity in nearest neighbors

def get_pos_tags(words):
    return pos_tag(words, lang='eng')

def print_nearest_neighbors_with_pos(word, word_vectors, vocab, top_n=10):
    if word not in vocab:
        print(f"{word} not found in vocabulary.")
        return

    word_idx = vocab.index(word)
    similarities = cosine_similarity([word_vectors[word_idx]], word_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1][1:top_n+1]
    
    neighbors = [vocab[i] for i in sorted_indices]
    
    all_words = [word] + neighbors
    pos_tags = get_pos_tags(all_words)
    
    print(f"Query word: {word}, POS: {pos_tags[0][1]}")
    print(f"Nearest neighbors and their POS tags:")
    for i, neighbor in enumerate(neighbors):
        print(f"{neighbor}: {pos_tags[i+1][1]}")

def run_pos_similarity():
    vocab_25k = load_vocab('./assn1/vocab-25k.txt')
    corpus = load_vocab('./wiki-1percent.txt/wiki-1percent.txt')

    for w in [1, 6]:
        C = build_word_context_matrix(corpus, vocab_25k, vocab_25k, window_size=w)
        Cpmi = compute_pmi(C)
        print(f"\nWindow Size {w}")
        print_nearest_neighbors_with_pos('monster', Cpmi, vocab_25k)

# ----------------------------------------------------------------------------------- #
# 1.4.3 Words with multiple senses

def run_multiple_sense_analysis():
    multi_sense_words = ['bank', 'cell', 'apple', 'light', 'well', 'frame', 'axes']
    
    vocab_25k = load_vocab('./assn1/vocab-25k.txt')
    corpus = load_vocab('./wiki-1percent.txt/wiki-1percent.txt')

    for w in [1, 6]:
        Cpmi = build_word_context_matrix(corpus, vocab_25k, vocab_25k, window_size=w)
        Cpmi = compute_pmi(Cpmi)
        
        print(f"\nWindow Size {w}")
        for word in multi_sense_words:
            print(f"\nNearest neighbors for '{word}' with window size {w}:")
            print_nearest_neighbors(word, Cpmi, vocab_25k)

# ----------------------------------------------------------------------------------- #
# 전체 실행 함수

def main():
    print("Running Distributional Counting...")
    run_distributional_counting()
    
    print("\nRunning PMI Calculation...")
    run_pmi()
    
    print("\nRunning Experimentation...")
    run_experimentation()
    
    print("\nRunning Nearest Neighbors Analysis...")
    run_nearest_neighbors()
    
    print("\nRunning POS Similarity Analysis...")
    run_pos_similarity()

    print("\nRunning Multiple Sense Word Analysis...")
    run_multiple_sense_analysis()
    
if __name__ == "__main__":
    main()
