# Purple AI V1.0 — model trenowany od zera

Purple AI to projekt małego modelu językowego trenowanego od zera oraz **API, z którego inne aplikacje mogą korzystać**. Nie ma tutaj strony czatu ani panelu do obserwowania treningu. Programista łączy swoją stronę, grę, aplikację lub bota z endpointem API, a API generuje tekst przy użyciu wag modelu Purple AI.

## Co znajduje się w repozytorium

- `training/train_from_scratch.py` — prototyp znakowego Transformera trenowanego od losowych wag.
- `training/data.example.txt` i `training/data.example.jsonl` — przykładowe formaty danych; nie są dużym zbiorem treningowym.
- `training/download_chat_data.py` — narzędzie pomocnicze do przygotowania danych; sprawdź licencje i zasady źródła przed użyciem.
- `api/app.py` — API HTTP, które ładuje lokalny checkpoint Purple AI.
- `api/README.md` — dokumentacja endpointów i przykłady użycia.
- `requirements-api.txt` — biblioteki potrzebne do uruchomienia API.

## Trening lokalny

Wymagany jest Python i PyTorch zgodny z komputerem. Przygotuj własny, legalnie używalny tekst w `training/data.txt` (nie dodawaj prywatnych danych ani dużych plików danych do publicznego repozytorium), następnie uruchom:

```bash
python -m pip install torch
python training/train_from_scratch.py
```

Model zapisuje checkpoint w `checkpoints/purple_ai_latest.pt`. To mały prototyp znakowy — jakość zależy od danych i treningu, nie jest to jeszcze odpowiednik dużego asystenta.

## Uruchomienie API (Windows CMD)

```bat
py -m pip install -r requirements-api.txt
set PURPLE_CHECKPOINT_PATH=checkpoints\purple_ai_latest.pt
set PURPLE_API_KEY=ustaw-tu-dlugi-losowy-sekret
py -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Test lokalny: `http://127.0.0.1:8000/docs`. Endpoint generowania to `POST /generate`; wymagany jest nagłówek `Authorization: Bearer TWOJ_KLUCZ`. Endpoint `/health` nie pokazuje ścieżek plików ani szczegółów wyjątków.

Przykład żądania:

```bash
curl -X POST http://127.0.0.1:8000/generate ^
  -H "Authorization: Bearer TWOJ_KLUCZ" ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Cześć\",\"max_new_chars\":120}"
```

## Ważne ograniczenia

- W repozytorium nie ma wytrenowanych wag; trzeba najpierw uruchomić trening i przechować checkpoint na serwerze, na którym działa API.
- Samo umieszczenie kodu na GitHubie nie uruchamia modelu ani nie hostuje API.
- API generuje tekst przy użyciu lokalnego checkpointu; nie korzysta z zewnętrznego dostawcy modelu.
- Nie publikuj kluczy API, prywatnych danych ani checkpointów w repozytorium.
- Przed wystawieniem API publicznie użyj HTTPS, ustaw mocny klucz i dodaj limity żądań po stronie hostingu/proxy.
