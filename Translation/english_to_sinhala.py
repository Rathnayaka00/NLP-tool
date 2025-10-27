"""
English-to-Sinhala Translation Module
Uses the 'thilina/mt5-sinhalese-english' model from Hugging Face Transformers.
"""

import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


# if __name__ == "__main__":
#     """
#     Main execution block to test the translation module.
#     This code only runs when the script is executed directly.
#     """
#     print("=== English-to-Sinhala Translation Test ===")

#     # --- Test Case 1 ---
#     text_1 = "Hello, how are you today?"
#     print(f"\nTesting with: '{text_1}'")
    
#     try:
#         # The first call will trigger the model loading
#         translation_1 = translate_en_to_si(text_1)
#         print(f"English: {text_1}")
#         print(f"Sinhala: {translation_1}")

#     except Exception as e:
#         print(f"Translation failed for Test 1: {e}", file=sys.stderr)
#         sys.exit(1) # Exit if the model failed to load

#     # --- Test Case 2 ---
#     text_2 = "My name is Gemini and I am a large language model."
#     print(f"\nTesting with: '{text_2}'")
    
#     try:
#         # This call should be fast as the model is already loaded
#         translation_2 = translate_en_to_si(text_2)
#         print(f"English: {text_2}")
#         print(f"Sinhala: {translation_2}")
        
#     except Exception as e:
#         print(f"Translation failed for Test 2: {e}", file=sys.stderr)

#     # --- Test Case 3 (Empty String) ---
#     text_3 = "   " # Test with whitespace only
#     print(f"\nTesting with whitespace: '{text_3}'")
    
#     try:
#         translation_3 = translate_en_to_si(text_3)
#         print(f"English: '{text_3}'")
#         print(f"Sinhala: '{translation_3}'  <-- (Should be empty)")
        
#     except Exception as e:
#         print(f"Translation failed for Test 3: {e}", file=sys.stderr)

#     print("\n=== Translation Test Complete ===")