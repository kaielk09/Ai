"""PurpleAI Windows helper: prepare public-domain Polish text, train, upload checkpoint."""
from __future__ import annotations

import getpass
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "training" / "train_from_scratch.py"
DATA = ROOT / "training" / "data.txt"
CHECKPOINT = ROOT / "checkpoints" / "purple_ai_latest.pt"
VOCAB = ROOT / "checkpoints" / "vocab.json"
REPO_ID = "Kamiljdds/purpleaiV1.0"


def install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def prepare_data() -> None:
    if DATA.exists() and DATA.stat().st_size > 10000:
        answer = input(f"\nZnaleziono {DATA} ({DATA.stat().st_size} bajtow).\nUzyc istniejacych danych? [T/n]: ").strip().lower()
        if answer not in ("n", "nie"):
            return

    try:
        import requests
    except ImportError:
        install("requests")
        import requests

    print("\nPobieram polskie teksty z API Wolnych Lektur...")
    response = requests.get("https://wolnelektury.pl/api/books/", timeout=40)
    response.raise_for_status()
    books = response.json()
    pieces: list[str] = []
    total_words = 0
    seen_urls = set()

    for book in books:
        if total_words >= 40000:
            break
        url = book.get("txt")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            text = r.content.decode("utf-8", errors="replace")
            text = re.sub(r"\s+", " ", text).strip()
            words = text.split()
            if len(words) < 100:
                continue
            words = words[:40000 - total_words]
            pieces.append(" ".join(words))
            total_words += len(words)
            print(f"  {book.get('title', 'Tekst')}: razem {total_words} slow")
            time.sleep(0.25)
        except Exception as exc:
            print(f"  Pomijam niedostepny tekst: {exc}")

    if total_words < 1000:
        raise RuntimeError("Pobrano za malo tekstu. Sprawdz internet i sprobuj ponownie.")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text("\n\n".join(pieces), encoding="utf-8")
    print(f"\nZapisano {total_words} slow do: {DATA}")


def main() -> None:
    if not TRAINER.exists():
        raise SystemExit(f"Nie znaleziono skryptu treningowego: {TRAINER}")

    prepare_data()
    print("\nUstawiam trening na 20 000 krokow, zapis co 100 krokow.")
    env = os.environ.copy()
    env["PURPLE_STEPS"] = "20000"
    env["PURPLE_EVAL_EVERY"] = "100"
    env["PURPLE_BATCH_SIZE"] = "16"

    print("\nRozpoczynam trening. Nie zamykaj tego okna CMD.")
    print("Checkpoint bedzie zapisywany lokalnie co 100 krokow.")
    subprocess.check_call([sys.executable, str(TRAINER)], cwd=str(ROOT), env=env)

    if not CHECKPOINT.exists():
        raise SystemExit("Trening sie zakonczyl, ale nie znaleziono checkpointu.")

    print("\nTrening zakonczony. Teraz mozna wyslac checkpoint do Hugging Face.")
    print("Token wpisz ponizej lokalnie. Nie bedzie wyswietlany ani zapisywany w pliku.")
    token = getpass.getpass("Wklej token Hugging Face (token z prawem zapisu): ").strip()
    if not token:
        print("Pominieto upload. Checkpoint zostal zachowany lokalnie.")
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
        print(f"\nUpload zakonczony: https://huggingface.co/{REPO_ID}")
    except Exception as exc:
        print(f"\nUpload nie powiodl sie: {exc}")
        print(f"Checkpoint nadal jest lokalnie: {CHECKPOINT}")
        print("Sprawdz token z prawem zapisu i dostep do repozytorium.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nProces zakonczyl sie bledem (kod {exc.returncode}).")
        print("Checkpointy zapisane wczesniej pozostaja na dysku.")
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"\nBLAD: {exc}")
        sys.exit(1)
