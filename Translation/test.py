import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = None
tokenizer = None
model_name = "thilina/mt5-sinhalese-english"


def _load_model():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return

    try:
        print(f"Loading model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Fatal error loading model: {e}", file=sys.stderr)
        print("Please check internet connection and 'transformers'/'torch' installation.", file=sys.stderr)
        model = None
        tokenizer = None
        raise e 


def translate_en_to_si(text_to_translate: str) -> str:
    global model, tokenizer

    try:
        if model is None or tokenizer is None:
            _load_model()

        if not text_to_translate or len(text_to_translate.strip()) == 0:
            return ""
        inputs = tokenizer(
            text_to_translate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

        translated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return translated_text

    except Exception as e:
        print(f"An error occurred during translation: {e}", file=sys.stderr)
        return ""


if __name__ == "__main__":
    print("=== English-to-Sinhala Translation Test ===")

    # --- Test Case 1 ---
    text_1 = "Hello, how are you today?"
    print(f"\nTesting with: '{text_1}'")
    
    try:
        # The first call will trigger the model loading
        translation_1 = translate_en_to_si(text_1)
        print(f"English: {text_1}")
        print(f"Sinhala: {translation_1}")

    except Exception as e:
        print(f"Translation failed for Test 1: {e}", file=sys.stderr)
        sys.exit(1) # Exit if the model failed to load

    # --- Test Case 2 ---
    text_2 = "My name is Gemini and I am a large language model."
    print(f"\nTesting with: '{text_2}'")
    
    try:
        # This call should be fast as the model is already loaded
        translation_2 = translate_en_to_si(text_2)
        print(f"English: {text_2}")
        print(f"Sinhala: {translation_2}")
        
    except Exception as e:
        print(f"Translation failed for Test 2: {e}", file=sys.stderr)

    # --- Test Case 3 (Empty String) ---
    text_3 = "   " # Test with whitespace only
    print(f"\nTesting with whitespace: '{text_3}'")
    
    try:
        translation_3 = translate_en_to_si(text_3)
        print(f"English: '{text_3}'")
        print(f"Sinhala: '{translation_3}'  <-- (Should be empty)")
        
    except Exception as e:
        print(f"Translation failed for Test 3: {e}", file=sys.stderr)

    print("\n=== Translation Test Complete ===")