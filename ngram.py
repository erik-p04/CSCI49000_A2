import re
from collections import defaultdict, Counter
import math
import random

#Tokenize the text into words ignoring punctuation
def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words

#Compute bigram probabilities
def build_bigram_model(corpus):
    bigrams = list(zip(corpus[:-1], corpus[1:]))
    unigram_counts = Counter(corpus)
    bigram_counts = Counter(bigrams)
    bigram_probabilities = defaultdict(float)
    for bigram, count in bigram_counts.items():
        bigram_probabilities[bigram] = count / unigram_counts[bigram[0]]
    
    return bigram_probabilities, bigram_counts

#print top 5 most probable bigrams
def top_bigrams(bigram_probabilities):
    sorted_bigrams = sorted(bigram_probabilities.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 most probable bigrams:")
    for bigram, prob in sorted_bigrams[:5]:
        print(f"{bigram}: {prob:.4f}")

#bigram probabilities using Laplace smoothing 
def laplace_smoothing(bigram_counts, unigram_counts, vocab_size):
    bigram_probabilities = defaultdict(float)
    for bigram, count in bigram_counts.items():
        bigram_probabilities[bigram] = (count + 1) / (unigram_counts[bigram[0]] + vocab_size)
    
    return bigram_probabilities

#compute perplexity of bigram model
def calculate_perplexity(test_corpus, bigram_probabilities):
    log_prob_sum = 0
    bigrams = list(zip(test_corpus[:-1], test_corpus[1:]))
    for bigram in bigrams:
        if bigram in bigram_probabilities:
            log_prob_sum += math.log2(bigram_probabilities[bigram])
        else:
            log_prob_sum += math.log2(1e-10)
    
    perplexity = 2 ** (-log_prob_sum / len(bigrams))
    return perplexity

#generate text using bigram model
def generate_text(start_word, bigram_probabilities, max_length=20):
    sentence = [start_word]
    current_word = start_word
    for _ in range(max_length - 1):
        next_words = [bigram[1] for bigram in bigram_probabilities if bigram[0] == current_word]
        if not next_words:
            break
        next_word = random.choice(next_words)
        sentence.append(next_word)
        current_word = next_word

    return ' '.join(sentence)

#calculate trigram probabilities
def build_trigram_model(corpus):
    trigrams = list(zip(corpus[:-2], corpus[1:-1], corpus[2:]))
    bigram_counts = Counter(zip(corpus[:-1], corpus[1:]))
    trigram_counts = Counter(trigrams)
    trigram_probabilities = defaultdict(float)
    for trigram, count in trigram_counts.items():
        bigram = trigram[:2]
        trigram_probabilities[trigram] = count / bigram_counts[bigram]

    return trigram_probabilities, trigram_counts

#print top 5 most probable trigrams
def top_trigrams(trigram_probabilities):
    sorted_trigrams = sorted(trigram_probabilities.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 most probable trigrams:")
    for trigram, prob in sorted_trigrams[:5]:
        print(f"{trigram}: {prob:.4f}")

def main():
    with open("corpus.txt", "r") as file:
        text = file.read()
    #tokenize the text
    tokens = tokenize(text)
    #Build the bigram model
    bigram_probabilities, bigram_counts = build_bigram_model(tokens)
    #print the top 5 bigrams (problem 4)
    top_bigrams(bigram_probabilities)
    #apply laplace smoothing (problem 5)
    unigram_counts = Counter(tokens)
    vocab_size = len(unigram_counts)
    smoothed_probabilities = laplace_smoothing(bigram_counts, unigram_counts, vocab_size)
    top_bigrams(smoothed_probabilities)
    #compute perplexity of bigram model on a test corpus (problem 6)
    perplexity = calculate_perplexity(tokens, bigram_probabilities)
    print(f"Perplexity: {perplexity:.4f}")
    #generate text (problem 7)
    start_word = random.choice(tokens)
    generated_text = generate_text(start_word, bigram_probabilities)
    print("Generated Text:")
    print(generated_text)
    #Build the trigram model (problem 8)
    trigram_probabilities, _ = build_trigram_model(tokens)
    top_trigrams(trigram_probabilities)

main()
