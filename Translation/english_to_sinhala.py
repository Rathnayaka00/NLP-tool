import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from nltk.tokenize import sent_tokenize

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

        # Helper: strip directive-like prefixes from output (both EN and Sinhala variants)
        def _clean_prefix(text: str) -> str:
            if not text:
                return text
            t = text.strip()
            patterns = [
                r"^(?i)translate\s+(the\s+following\s+)?english\s+to\s+(sinhala|sinhalese)\s*:\s*",
                r"^(?i)english\s*:\s*",
                r"^(?i)sinhala\s*:\s*",
            ]
            for pat in patterns:
                t = re.sub(pat, "", t).strip()
            # Sinhala directive heuristic: if early colon and directive words present, drop up to colon
            if ("පරිවර්තනය" in t[:40] or "සිංහල" in t[:40]) and ":" in t[:40]:
                idx = t.find(":")
                if 0 <= idx <= 40:
                    t = t[idx+1:].lstrip()
            return t

        # Helper: translate a single chunk; try plain first, fallback to prefixed
        def _translate_chunk(chunk: str) -> str:
            # Plain input
            plain_inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            outputs = model.generate(
                **plain_inputs,
                max_new_tokens=400,
                num_beams=6,
                length_penalty=1.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned = _clean_prefix(decoded)
            # If the output remains English (no Sinhala unicode), fallback with prefixed prompt
            if (not cleaned) or (re.search(r"[A-Za-z]", cleaned) and not re.search(r"[\u0D80-\u0DFF]", cleaned)) or cleaned.strip() == chunk.strip():
                prefixed = f"translate English to Sinhala: {chunk}"
                prefixed_inputs = tokenizer(
                    prefixed,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                outputs2 = model.generate(
                    **prefixed_inputs,
                    max_new_tokens=400,
                    num_beams=6,
                    length_penalty=1.0,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                decoded2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
                cleaned2 = _clean_prefix(decoded2)
                return cleaned2
            return cleaned

        # Preserve paragraphs and translate sentence-by-sentence to avoid models that only translate the first sentence
        paragraphs = [p for p in text_to_translate.split('\n\n') if p.strip()]
        translated_paragraphs = []
        for para in paragraphs:
            # Robust sentence splitting with fallback (keeps punctuation)
            def _split_into_sentences(text: str):
                try:
                    return sent_tokenize(text)
                except Exception:
                    pass
                parts = re.findall(r"[^.!?\n]+[.!?]+|[^.!?\n]+$", text)
                parts = [p.strip() for p in parts if p and p.strip()]
                if parts:
                    return parts
                # Last resort: split by ~25 words
                words = text.split()
                chunk_size = 25
                return [" ".join(words[i:i+chunk_size]).strip() for i in range(0, len(words), chunk_size) if " ".join(words[i:i+chunk_size]).strip()]

            sentences = _split_into_sentences(para)

            translated_sentences = []
            for s in sentences:
                translated_sentences.append(_translate_chunk(s))
            translated_paragraphs.append(' '.join(translated_sentences))

        return '\n\n'.join(translated_paragraphs)

    except Exception as e:
        print(f"An error occurred during translation: {e}", file=sys.stderr)
        return ""