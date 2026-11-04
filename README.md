# Purple AI V1.0

**Cel projektu:** jeden autorski model trenowany od losowych wag i jedno API rozmowy, z którego mogą korzystać różne programy.

## Jedna AI, wiele klientów

CMD, PowerShell, strony internetowe, aplikacje, gry i boty Discord są klientami. Wszystkie łączą się z tym samym endpointem `POST /v1/chat`. Nie tworzymy osobnego modelu dla każdego rodzaju programu.

## Co jest w repozytorium

- `training/train_from_scratch.py` — mały znakowy Transformer trenowany od losowych wag.
- `training/data.example.txt` i `training/data.example.jsonl` — przykładowe formaty danych, nie duży zbiór treningowy.
- `training/download_chat_data.py` — narzędzie pomocnicze do przygotowania danych; przed użyciem sprawdzaj licencje i zasady źródła.
- `api/app.py` — serwer API z jednym endpointem rozmowy `POST /v1/chat`.
- `api/README.md` — instrukcja uruchomienia i podłączania klientów.

## Trening od zera

Wymagany jest Python i PyTorch zgodny z komputerem. Przygotuj legalnie używalne dane tekstowe w `training/data.txt`, a następnie uruchom:

```bash
python -m pip install torch
python training/train_from_scratch.py
```

Checkpoint zostanie zapisany w `checkpoints/purple_ai_latest.pt`. To mały prototyp edukacyjny, a nie gotowy, zaawansowany asystent.

## Uruchomienie API

Instrukcja Windows znajduje się w [api/README.md](api/README.md). API wymaga wytrenowanego checkpointu i sekretu `PURPLE_API_KEY`. Opcjonalne wyszukiwanie wymaga serwerowego sekretu `SEARCH_API_KEY`.

## Stan projektu — bez ukrywania ograniczeń

- Kod treningowy modelu od losowych wag: istnieje.
- Jeden endpoint rozmowy dla różnych klientów: dodany.
- Wytrenowane wagi modelu: **nie ma ich w repozytorium**.
- Jakość zaawansowanego asystenta: **nieosiągnięta**.
- Wdrożenie produkcyjne i testy obciążeniowe: **do wykonania**.

GitHub przechowuje kod, ale sam nie uruchamia modelu ani nie hostuje API. Do działania potrzebny jest serwer z odpowiednim sprzętem i zapisanym checkpointem.
