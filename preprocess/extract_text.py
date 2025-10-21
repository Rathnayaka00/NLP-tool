import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from llama_parse import LlamaParse


def _init_parser() -> LlamaParse:
	load_dotenv(override=False)
	return LlamaParse(result_type="markdown")


def parse_pdf_to_markdown(file_path: str) -> str:
	p = Path(file_path)
	if not p.exists():
		raise FileNotFoundError(f"File not found: {file_path}")
	if p.suffix.lower() != ".pdf":
		raise ValueError("Only PDF files are supported for parsing.")

	parser = _init_parser()
	extra_info = {"file_name": p.name}

	with p.open("rb") as f:
		documents = parser.load_data(f, extra_info=extra_info)

	parts: List[str] = []
	for doc in documents:
		text = getattr(doc, "text", None) or getattr(doc, "get_content", lambda: "")()
		if text:
			parts.append(text)

	return "\n\n".join(parts)

