"""PurpleAI Windows helper: download Polish public-domain texts, train, upload checkpoint."""
from __future__ import annotations

import getpass
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "training" / "train_from_scratch.py"
DATA = ROOT / "training" / "data.txt"
CHECKPOINT = ROOT / "checkpoints" / "purple_ai_latest.pt"
VOCAB = ROOT / "checkpoints" / "vocab.json"
REPO_ID = "Kamiljdds/purpleaiV1.0"
API = "https://wolnelektury.pl/api"


def install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def prepare_data() -> None:
    try:
        import requests
    except ImportError:
        install("requests")
        import requests

    print("\nPobieram liste ksiazek z oficjalnego API Wolnych Lektur...")
    session = requests.Session()
    session.headers.update({"User-Agent": "PurpleAI-learning-project/1.0"})
    response = session.get(f"{API}/books/?format=json", timeout=40)
    response.raise_for_status()
    books = response.json()
    if not isinstance(books, list) or not books:
        raise RuntimeError("API nie zwrocilo listy ksiazek.")

    pieces: list[str] = []
    total_words = 0
    seen = set()
    attempted = 0

    for book in books:
        if total_words >= 40000:
            break
        # The list endpoint may not include a direct txt URL.
        detail_url = book.get("href") or book.get("api_url")
        if not detail_url:
            slug = book.get("slug")
            if not slug and book.get("url"):
                slug = urlparse(book["url"]).path.rstrip("/").split("/")[-1]
            if slug:
                detail_url = f"{API}/books/{slug}/"
        if not detail_url or detail_url in seen:
            continue
        seen.add(detail_url)
        attempted += 1
        try:
            detail_response = session.get(detail_url, timeout=30)
            if detail_response.status_code != 200:
                continue
            detail = detail_response.json()
            language = str(detail.get("language", "pol")).lower()
            if language not in ("pol", "pl", "polski"):
                continue
            txt_url = detail.get("txt_url") or detail.get("txt")
            if not txt_url:
                continue
            text_response = session.get(txt_url, timeout=40)
            if text_response.status_code != 200:
                continue
            text = text_response.content.decode("utf-8", errors="replace")
            text = re.sub(r"\r\n?", "\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            words = text.split()
            if len(words) < 100:
                continue
            words = words[:40000 - total_words]
            pieces.append(
                f"\n\n[Zrodlo: Wolne Lektury | Autor: {detail.get('author', book.get('author', 'nieznany'))} | "
                f"Tytul: {detail.get('title', book.get('title', 'nieznany'))}]\n"
                + " ".join(words)
            )
            total_words += len(words)
            print(f"  {detail.get('title', book.get('title', 'tekst'))}: {total_words}/40000 slow")
            time.sleep(0.15)
        except Exception as exc:
            print(f"  Pomijam jeden tekst ({type(exc).__name__}).")

    if total_words < 1000:
        raise RuntimeError(
            f"API odpowiedzialo, ale zebrano tylko {total_words} slow po sprawdzeniu {attempted} rekordow. "
            "Nie uruchamiam treningu z tak mala iloscia danych."
        )
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text("\n".join(pieces), encoding="utf-8")
    print(f"\nGotowe: {total_words} slow zapisano do:\n{DATA}")


def main() -> None:
    if not TRAINER.exists():
        raise SystemExit(f"Nie znaleziono skryptu treningowego: {TRAINER}")
    prepare_data()

    print("\nTrening: 20 000 krokow; checkpoint co 100 krokow.")
    env = os.environ.copy()
    env["PURPLE_STEPS"] = "20000"
    env["PURPLE_EVAL_EVERY"] = "100"
    env["PURPLE_BATCH_SIZE"] = "16"
    subprocess.check_call([sys.executable, str(TRAINER)], cwd=str(ROOT), env=env)

    if not CHECKPOINT.exists():
        raise SystemExit("Trening sie zakonczyl, ale nie znaleziono pliku checkpointu.")

    print("\nTrening zakonczony. Checkpoint zapisany lokalnie.")
    token = getpass.getpass("Wklej token Hugging Face z prawem zapisu (ukryty podczas wpisywania): ").strip()
    if not token:
        print(f"Pominieto upload. Plik pozostaje tutaj: {CHECKPOINT}")
        return

    try:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            install("huggingface_hub")
            from huggingface_hub import HfApi
        api = HfApi(token=token)
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
        print(f"Upload zakonczony: https://huggingface.co/{REPO_ID}")
    except Exception as exc:
        print(f"Upload nie powiodl sie: {exc}")
        print(f"Checkpoint nadal jest lokalnie: {CHECKPOINT}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nProces zakonczyl sie bledem (kod {exc.returncode}). Wczesniejsze checkpointy pozostaja na dysku.")
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"\nBLAD: {exc}")
        sys.exit(1)
