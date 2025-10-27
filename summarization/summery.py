import nltk
import re
import heapq
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import math


for resource in [
    "punkt",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",     
    "averaged_perceptron_tagger_eng", 
    "omw-1.4",
]:
    try:
        if resource in ["punkt"]:
            nltk.data.find(f"tokenizers/{resource}")
        elif resource in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
            nltk.data.find(f"taggers/{resource}")
        else:
            nltk.data.find(f"corpora/{resource}")
    except LookupError:
        print(f"Downloading missing NLTK resource: {resource}")
        nltk.download(resource)


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def summarize_text(text: str, num_sentences: int = 5) -> str:
    try:
        if not text or len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty.")

        if not isinstance(num_sentences, int) or num_sentences <= 0:
            raise ValueError("Number of sentences must be a positive integer.")

        clean_text = re.sub(r"\s+", " ", text)
        clean_text = re.sub(r"\[[0-9]*\]", " ", clean_text)
        clean_text = re.sub(r"[^a-zA-Z0-9.,!?-]", " ", clean_text)

        sentences = sent_tokenize(text)

        if len(sentences) <= 1:
            words = text.split()
            chunk_size = 25
            sentences = [
                " ".join(words[i:i + chunk_size]).strip() + "."
                for i in range(0, len(words), chunk_size)
            ]

        if len(sentences) <= num_sentences:
            return text

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

        word_frequencies = {}
        for word in lemmatized_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

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

        tf_idf_scores = {
            word: word_frequencies[word] * idf.get(word, 0) for word in word_frequencies}

        sentence_scores = {}
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for word, score in tf_idf_scores.items():
                if word in [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokenize(sentence_lower)]:
                    sentence_scores[sentence] = sentence_scores.get(
                        sentence, 0) + score

        summary_sentences = heapq.nlargest(
            num_sentences, sentence_scores, key=sentence_scores.get)

        summary = " ".join(sentence.strip() for sentence in summary_sentences)
        return summary

    except LookupError as e:
        print(
            f"NLTK resource error: {e}. Ensure all required resources are downloaded.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

