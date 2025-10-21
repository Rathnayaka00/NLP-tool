import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_data():

    print("Initializing NLTK data...")
    required_packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    
    for package in required_packages:
        try:
            if package == 'punkt' or package == 'punkt_tab':
                nltk.data.find(f'tokenizers/{package}')
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