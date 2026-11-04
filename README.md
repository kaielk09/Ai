# Purple AI V1.0

Cel: jeden autorski model trenowany od losowych wag i jedno API rozmowy dla wielu rodzajów programów.

## Jeden model — wiele klientów

CMD, PowerShell, strony WWW, aplikacje mobilne i desktopowe, gry oraz boty Discord są klientami tego samego API. Wszystkie wysyłają historię rozmowy do `POST /v1/chat`. Nie tworzymy osobnego modelu dla każdego klienta.

## Zawartość

- `training/train_from_scratch.py` — własna implementacja małego Transformera trenowanego od losowych wag.
- Tokenizer bajtowy UTF-8 — 256 identyfikatorów bajtów; potrafi reprezentować polskie znaki i Unicode.
- `training/data.example.txt` i `training/data.example.jsonl` — przykładowe formaty, nie gotowy duży zbiór treningowy.
- `api/app.py` — jeden endpoint rozmowy `POST /v1/chat`.
- `api/README.md` — instrukcje treningu, uruchomienia i integracji.

## Trening

Przygotuj legalnie używalny tekst w lokalnym `training/data.txt`, a następnie:

```bash
python -m pip install torch
python training/train_from_scratch.py
```

Checkpoint jest zapisywany do `checkpoints/purple_ai_latest.pt`. Szczegóły i ustawienia opisano w [training/README.md](training/README.md) oraz [api/README.md](api/README.md).

## API

Uruchom serwer zgodnie z [api/README.md](api/README.md). Każdy klient wysyła żądanie do `POST /v1/chat`. Wyszukiwanie internetowe może być włączone parametrem `web_search`; jest narzędziem API, nie zewnętrznym modelem AI.

## Uczciwy status

- Kod treningowy od zera: jest.
- Tokenizer bajtowy UTF-8: dodany.
- Jeden endpoint rozmowy dla różnych klientów: jest.
- Wytrenowane wagi: nie ma ich w repozytorium — trzeba wykonać trening.
- Zaawansowane rozumowanie i jakość dużego asystenta: nie są jeszcze osiągnięte.
- Testy na gotowym checkpointcie i wdrożenie produkcyjne: do wykonania.

GitHub przechowuje kod, ale sam nie trenuje modelu ani nie hostuje API. Do tego potrzebny jest komputer lub serwer z odpowiednim sprzętem.
