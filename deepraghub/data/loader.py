# deepraghub/data/loader.py
from typing import List, Dict
from pathlib import Path


def load_documents(directory: str) -> List[Dict[str, str]]:
    documents = []
    for file_path in Path(directory).rglob("*"):
        if file_path.is_file() and file_path.suffix in [".txt", ".md", ".pdf"]:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(
                    {
                        "content": content,
                        "source": str(file_path),
                        "type": file_path.suffix[1:],
                    }
                )
    return documents
