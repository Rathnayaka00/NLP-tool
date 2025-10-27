import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_data():

    print("Initializing NLTK data...")
    # Include both legacy and language-specific taggers to support NLTK >= 3.8
    required_packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger',           # legacy name
        'averaged_perceptron_tagger_eng'        # new name in newer NLTK
    ]
    
    for package in required_packages:
        try:
            if package in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{package}')
            elif package in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
                nltk.data.find(f'taggers/{package}')
            else:
                nltk.data.find(f'corpora/{package}')
            print(f"NLTK package '{package}' is already downloaded.")
        except LookupError:
            print(f"NLTK package '{package}' not found. Downloading...")
            nltk.download(package, quiet=True)
            print(f"NLTK package '{package}' downloaded successfully.")
    print("NLTK data is ready.")

def preprocess_text_for_summarization(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        token for token in tokens
        if token.isalpha() and token not in stop_words
    ]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    cleaned_text = ' '.join(lemmatized_tokens)

    return cleaned_text