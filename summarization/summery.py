"""
Text Summarization Module
Summarizes English text using TF-IDF algorithm with Lemmatization (handles missing punctuation)
"""

import nltk
import re
import heapq
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import math


for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "omw-1.4"]:
    try:
        # Define the correct path for each type of resource
        if resource in ["punkt"]:
            nltk.data.find(f"tokenizers/{resource}")
        elif resource in ["averaged_perceptron_tagger"]:
            nltk.data.find(f"taggers/{resource}")  # <-- Corrected path
        else:
            nltk.data.find(f"corpora/{resource}")
    except LookupError:
        print(f"Downloading missing NLTK resource: {resource}")
        nltk.download(resource)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def summarize_text(text: str, num_sentences: int = 5) -> str:
    """
    Summarize English text using TF-IDF algorithm with lemmatization.
    Handles text even if it lacks punctuation.

    Args:
        text (str): Input text to summarize.
        num_sentences (int): Number of sentences to include in summary.

    Returns:
        str: Summarized text.
    """
    try:
        if not text or len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty.")

        if not isinstance(num_sentences, int) or num_sentences <= 0:
            raise ValueError("Number of sentences must be a positive integer.")

        # Clean text
        clean_text = re.sub(r"\s+", " ", text)
        clean_text = re.sub(r"\[[0-9]*\]", " ", clean_text)
        clean_text = re.sub(r"[^a-zA-Z-]", " ", clean_text)

        # Try normal sentence tokenization first
        sentences = sent_tokenize(text)

        # Fallback: if text has no punctuation, split roughly every 20–25 words
        if len(sentences) <= 1:
            words = text.split()
            chunk_size = 25
            sentences = [
                " ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)
            ]

        if len(sentences) <= num_sentences:
            return text

        # Tokenize words, remove stopwords, and lemmatize
        words = word_tokenize(clean_text.lower())
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        lemmatized_words = [
            lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in words
            if w not in stop_words and len(w) > 1
        ]

        if not lemmatized_words:
            raise ValueError("No valid words found after preprocessing.")

        # Compute Term Frequency (TF)
        word_frequencies = {}
        for word in lemmatized_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

        # Compute Inverse Document Frequency (IDF)
        idf = {}
        total_sentences = len(sentences)
        for sentence in sentences:
            sent_words = [lemmatizer.lemmatize(
                w.lower(), get_wordnet_pos(w)) for w in word_tokenize(sentence)]
            unique_words = set(sent_words)
            for word in unique_words:
                if word in word_frequencies:
                    idf[word] = idf.get(word, 0) + 1

        for word in idf:
            idf[word] = math.log(total_sentences / (idf[word] + 1))

        # Compute TF-IDF scores for words
        tf_idf_scores = {
            word: word_frequencies[word] * idf.get(word, 0) for word in word_frequencies}

        # Score each sentence
        sentence_scores = {}
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for word, score in tf_idf_scores.items():
                if word in [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokenize(sentence_lower)]:
                    sentence_scores[sentence] = sentence_scores.get(
                        sentence, 0) + score

        # Select top N sentences
        summary_sentences = heapq.nlargest(
            num_sentences, sentence_scores, key=sentence_scores.get)

        summary = " ".join(summary_sentences)
        return summary

    except LookupError as e:
        print(
            f"NLTK resource error: {e}. Ensure all required resources are downloaded.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


# Example usage
# if __name__ == "__main__":
#     test_text = """
#     Machine learning, a cornerstone of modern artificial intelligence,
#     is the scientific study of algorithms and statistical models that computer
#     systems use to perform a specific task without using explicit instructions, 
#     relying on patterns and inference instead. It is seen as a subset of artificial intelligence.
#     Machine learning algorithms build a mathematical model based on sample data, known as "training data",
#     in order to make predictions or decisions without being explicitly programmed to perform the task.
#     These algorithms are used in a wide variety of applications, such as email filtering and computer vision,
#     where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks.

#     There are several major categories of machine learning. In supervised learning, the algorithm is provided
#     with a pre-labeled dataset, meaning each data point is tagged with the correct output. The goal is for the
#     algorithm to learn a general rule that maps inputs to outputs, which can then be used to predict outcomes for new,
#     unseen data. In contrast, unsupervised learning involves working with data that has not been labeled. The algorithm
#     must explore the data and find structure on its own, such as grouping data points into clusters or reducing dimensionality to find the most important features.

#     A third prominent category is reinforcement learning, an area concerned with how software agents ought to take actions in an environment
#     in order to maximize some notion of cumulative reward. This type of learning is goal-oriented and is often used in robotics, gaming, and navigation.
#     Furthermore, deep learning, a subfield of machine learning, utilizes neural networks with many layers (hence "deep") to model and understand complex patterns
#     in large amounts of data. Deep learning has been the driving force behind recent breakthroughs in image recognition, natural language processing, and autonomous vehicles.
#     The continuous evolution of these techniques promises even more powerful applications in the future.
#     """
#     summary = summarize_text(test_text, num_sentences=5)
#     print("Summary:\n", summary)
