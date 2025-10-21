"""
English-to-Sinhala Translation Module
Uses the 'thilina/mt5-sinhalese-english' model from Hugging Face Transformers.
"""

import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Module-level globals ---
# We define them as None initially.
# They will be loaded by _load_model() on the first function call.
model = None
tokenizer = None
model_name = "thilina/mt5-sinhalese-english"


def _load_model():
    """
    Internal function to lazily load the model and tokenizer.
    This is only called once, the first time translation is requested.
    """
    global model, tokenizer
    
    # Check if already loaded (in case of concurrent calls, though this isn't thread-safe)
    if model is not None and tokenizer is not None:
        return

    try:
        print(f"Loading model: {model_name}...")
        # 1. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 2. Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Fatal error loading model: {e}", file=sys.stderr)
        print("Please check internet connection and 'transformers'/'torch' installation.", file=sys.stderr)
        # Set to None so we can retry, or re-raise to stop execution
        model = None
        tokenizer = None
        raise e  # Re-raise the exception so the caller knows loading failed


def translate_en_to_si(text_to_translate: str) -> str:
    """
    Translates a given English text string to Sinhala using the MT5 model.
    Lazily loads the model on the first call.

    Args:
        text_to_translate (str): The English text to translate.

    Returns:
        str: The translated Sinhala text, or an empty string on error.
    """
    global model, tokenizer

    try:
        # 1. Load the model if it's not already loaded
        if model is None or tokenizer is None:
            _load_model()
            
        # 2. Check for empty input
        if not text_to_translate or len(text_to_translate.strip()) == 0:
            return ""

        # 3. Tokenize the input text
        inputs = tokenizer(
            text_to_translate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # 4. Generate the translation
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

        # 5. Decode the generated tokens
        translated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return translated_text

    except Exception as e:
        print(f"An error occurred during translation: {e}", file=sys.stderr)
        return ""


# --- Example Usage ---
# if __name__ == "__main__":
#     english_text = "Hansaka is a fucking playboy and a foolish man."
# 
#     print(f"\nTranslating text...")
#     print(f"English: {english_text}")
# 
#     # Get the translation
#     # This will trigger the model download on the first run
#     sinhala_translation = translate_en_to_si(english_text)
# 
#     print(f"Sinhala: {sinhala_translation}")