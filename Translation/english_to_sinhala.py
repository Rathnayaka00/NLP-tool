from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Define the model name
model_name = "thilina/mt5-sinhalese-english"

# 2. Load the tokenizer and model
# This might take a moment the first time you run it
# as it needs to download the model (over 1GB).
try:
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection and if 'transformers' is installed.")
    exit()


def translate_en_to_si(text_to_translate):
    """
    Translates a given English text string to Sinhala.
    """
    if not text_to_translate:
        return ""

    try:
        # 3. Tokenize the input text
        # We prepare the text for the model
        inputs = tokenizer(
            text_to_translate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Max length for a single input
        )

        # 4. Generate the translation
        # The model generates the token IDs for the translated text
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,  # Use beam search for better quality
            early_stopping=True
        )

        # 5. Decode the generated tokens
        # Convert the token IDs back into a readable string
        translated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return translated_text

    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return ""


# --- Example Usage ---
if __name__ == "__main__":
    english_text = "Hansaka is a fucking playboy and a foolish man."

    print(f"\nTranslating text...")
    print(f"English: {english_text}")

    # Get the translation
    sinhala_translation = translate_en_to_si(english_text)

    print(f"Sinhala: {sinhala_translation}")
