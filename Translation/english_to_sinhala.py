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