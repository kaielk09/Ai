"""Pobiera prawdziwe polskie przykłady instrukcji i dopisuje je do training/data.txt.
Uruchom z katalogu głównego repozytorium:
    py -3.12 training\\download_chat_data.py
Wymaga: py -3.12 -m pip install -U datasets
"""
from pathlib import Path
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Brakuje biblioteki datasets.")
    print("Uruchom: py -3.12 -m pip install -U datasets")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data.txt"
MARKER = "\\n\\n===== PURPLEAI CHAT DATA =====\\n"
TARGET = 100_000

# Rzeczywiste zbiory publicznie opisane jako polskie dane instruktażowe.
# Trzeci jest używany tylko wtedy, gdy pierwsze dwa nie wystarczą.
SOURCES = [
    ("JohnTdi/polish-llm-sft-pl", "train"),
    ("pelcra/PLLuMIC-syn-ext", "train"),
    ("cpral/polskie-sprawy-v3", "train"),
]


def get_pairs(row):
    pairs = []
    messages = row.get("messages")

    if isinstance(messages, list):
        last_user = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role in ("user", "human"):
                last_user = content
            elif role in ("assistant", "bot") and last_user:
                pairs.append((last_user, content))
                last_user = None
        if pairs:
            return pairs

    # Obsługa popularnych formatów instruction/input/output oraz prompt/response.
    question = (
        row.get("instruction")
        or row.get("question")
        or row.get("prompt")
        or row.get("user")
    )
    answer = (
        row.get("output")
        or row.get("answer")
        or row.get("response")
        or row.get("assistant")
    )
    extra = row.get("input")

    if isinstance(question, str) and isinstance(answer, str):
        question = question.strip()
        answer = answer.strip()
        if extra and isinstance(extra, str) and extra.strip():
            question += "\\nKontekst: " + extra.strip()
        if question and answer:
            return [(question, answer)]

    return []


def main():
    # Przy kolejnym uruchomieniu zachowujemy książki i zastępujemy starą
    # sekcję rozmów, żeby nie dopisywać tych samych danych ponownie.
    old = DATA_FILE.read_text(encoding="utf-8") if DATA_FILE.exists() else ""
    book_text = old.split(MARKER, 1)[0].rstrip()

    unique = {}
    for repo, split in SOURCES:
        if len(unique) >= TARGET:
            break

        print(f"\\nPobieram przykłady z: {repo}", flush=True)
        try:
            dataset = load_dataset(repo, split=split, streaming=True)
            seen = 0
            for row in dataset:
                seen += 1
                for question, answer in get_pairs(row):
                    question = question.strip()
                    answer = answer.strip()
                    if len(question) < 2 or len(answer) < 2:
                        continue
                    key = (question.casefold(), answer.casefold())
                    unique.setdefault(key, (question, answer))
                    if len(unique) >= TARGET:
                        break
                if len(unique) >= TARGET:
                    break
                if seen % 5000 == 0:
                    print(f"  Przetworzono {seen:,}; unikalnych par: {len(unique):,}".replace(",", " "), flush=True)
            print(f"  Koniec źródła. Unikalnych par łącznie: {len(unique):,}".replace(",", " "))
        except Exception as error:
            print(f"  Nie udało się użyć {repo}: {error}")
            print("  Przechodzę do następnego źródła.")

    if not unique:
        print("\\nNie pobrano żadnych par. Plik data.txt pozostał bez zmian.")
        sys.exit(1)

    chat_lines = []
    for question, answer in unique.values():
        chat_lines.append(f"Użytkownik: {question}\\nPurpleAI: {answer}")

    combined = book_text + MARKER + "\\n\\n".join(chat_lines) + "\\n"
    DATA_FILE.write_text(combined, encoding="utf-8")

    print("\\nGOTOWE")
    print(f"Liczba unikalnych par pytań i odpowiedzi: {len(unique):,}".replace(",", " "))
    print(f"Zapisano razem z książkami do: {DATA_FILE}")
    if len(unique) < TARGET:
        print(f"Nie osiągnięto 100 000; brakuje {TARGET - len(unique):,} par.".replace(",", " "))
    print("\\nUwaga: obecny train_from_scratch.py uczy znak po znaku na data.txt;")
    print("to nie jest jeszcze pełny trening czatowy i nie gwarantuje jakości ChatGPT.")


if __name__ == "__main__":
    main()
