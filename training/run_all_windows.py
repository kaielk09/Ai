"""PurpleAI Windows helper: download Polish public-domain texts, train 20k steps, upload to HF."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "training" / "train_from_scratch.py"
DATA = ROOT / "training" / "data.txt"
CHECKPOINT = ROOT / "checkpoints" / "purple_ai_latest.pt"
VOCAB = ROOT / "checkpoints" / "vocab.json"
REPO_ID = "Kamiljdds/purpleaiV1.0"
API = "https://wolnelektury.pl/api"
TARGET_WORDS = 40000
TRAIN_STEPS = 20000


def install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def prepare_data() -> None:
    try:
        import requests
    except ImportError:
        install("requests")
        import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "PurpleAI-learning-project/1.1"})
    print("\nPobieram katalog Wolnych Lektur...")
    response = session.get(f"{API}/books/?format=json", timeout=60)
    response.raise_for_status()
    books = response.json()
    if isinstance(books, dict):
        books = books.get("results") or books.get("books") or []
    if not isinstance(books, list) or not books:
        raise RuntimeError("API nie zwrocilo listy ksiazek w oczekiwanym formacie.")

    pieces: list[str] = []
    total_words = 0
    attempted = 0
    downloaded = 0
    seen = set()

    for book in books:
        if total_words >= TARGET_WORDS:
            break
        slug = book.get("slug")
        if not slug and book.get("url"):
            slug = urlparse(book["url"]).path.rstrip("/").split("/")[-1]
        detail_url = book.get("api_url")
        if not detail_url or "/api/books/" not in detail_url:
            detail_url = f"{API}/books/{slug}/" if slug else None
        if not detail_url:
            continue
        detail_url = urljoin(API + "/", detail_url)
        if detail_url in seen:
            continue
        seen.add(detail_url)
        attempted += 1

        try:
            dr = session.get(detail_url, timeout=35)
            dr.raise_for_status()
            detail = dr.json()
            if isinstance(detail, dict) and "book" in detail and isinstance(detail["book"], dict):
                detail = detail["book"]
            language = str(detail.get("language", "pol")).lower()
            if language not in ("pol", "pl", "polski"):
                continue
            txt_url = detail.get("txt_url") or detail.get("txt")
            if not txt_url:
                continue
            txt_url = urljoin(detail_url, txt_url)
            tr = session.get(txt_url, timeout=60)
            tr.raise_for_status()
            text = tr.content.decode("utf-8-sig", errors="replace")
            text = re.sub(r"\r\n?", "\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            words = text.split()
            if len(words) < 100:
                continue
            words = words[:TARGET_WORDS - total_words]
            title = detail.get("title") or book.get("title") or slug or "nieznany"
            author = detail.get("author") or detail.get("author_names") or book.get("author") or "nieznany"
            if isinstance(author, list):
                author = ", ".join(str(x) for x in author)
            source_url = detail.get("url") or book.get("url") or detail_url
            pieces.append(
                f"\n\n[Źródło: Wolne Lektury | Autor: {author} | Tytuł: {title} | URL: {source_url}]\n"
                + " ".join(words)
            )
            total_words += len(words)
            downloaded += 1
            print(f"  Pobrano {downloaded} tekstów; łącznie {total_words}/{TARGET_WORDS} słów — {title}")
            time.sleep(0.12)
        except Exception as exc:
            print(f"  Pomijam rekord {slug or '?'} ({type(exc).__name__}: {exc})")

    if total_words < 5000:
        raise RuntimeError(
            f"Pobrano tylko {total_words} słów z {attempted} rekordów. "
            "Nie uruchamiam treningu; sprawdź połączenie/API."
        )
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text("\n".join(pieces), encoding="utf-8")
    print(f"\nGotowe: {total_words} słów z {downloaded} tekstów zapisano w {DATA}")


def upload_model() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        install("huggingface_hub")
        from huggingface_hub import HfApi

    api = HfApi()
    print("\nSprawdzam logowanie Hugging Face...")
    who = api.whoami()
    print("Zalogowano jako:", who.get("name") or who.get("fullname") or "użytkownik")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(CHECKPOINT),
        path_in_repo="purple_ai_latest.pt",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload PurpleAI training checkpoint",
    )
    if VOCAB.exists():
        api.upload_file(
            path_or_fileobj=str(VOCAB),
            path_in_repo="vocab.json",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload PurpleAI vocabulary",
        )
    print(f"\nSUKCES — model i słownik wysłane: https://huggingface.co/{REPO_ID}")


def main() -> None:
    if not TRAINER.exists():
        raise SystemExit(f"Nie znaleziono skryptu treningowego: {TRAINER}")
    prepare_data()
    print(f"\nTrening od zera: {TRAIN_STEPS:,} kroków; checkpoint co 100 kroków.")
    env = os.environ.copy()
    env["PURPLE_STEPS"] = str(TRAIN_STEPS)
    env["PURPLE_EVAL_EVERY"] = "100"
    env["PURPLE_BATCH_SIZE"] = "16"
    subprocess.check_call([sys.executable, str(TRAINER)], cwd=str(ROOT), env=env)
    if not CHECKPOINT.exists() or not VOCAB.exists():
        raise SystemExit("Trening się zakończył, ale nie znaleziono checkpointu lub vocab.json.")
    upload_model()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nProces zakończył się błędem (kod {exc.returncode}); wcześniejsze checkpointy zostają na dysku.")
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"\nBŁĄD: {exc}")
        sys.exit(1)
