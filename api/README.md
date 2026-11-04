# Purple AI V1.0 — jedno API rozmowy

Wszystkie programy korzystają z jednego endpointu: `POST /v1/chat`. Może to być CMD, PowerShell, strona, aplikacja, gra lub bot Discord. Klient wysyła historię rozmowy do API; wszystkie klienty używają tych samych wag modelu.

## Model i checkpoint

Model jest trenowany od losowych wag. Używa własnej implementacji małego Transformera i tokenizera bajtowego UTF-8 (256 możliwych wartości bajtu). Dzięki temu tokenizer może reprezentować polskie znaki i inne znaki Unicode bez pomijania ich ze słownika.

Repozytorium nie zawiera wytrenowanych wag. Przed uruchomieniem API trzeba wytrenować model i uzyskać plik `checkpoints/purple_ai_latest.pt`.

## Trening (Windows)

Przygotuj własne, legalnie używalne dane tekstowe w pliku `training/data.txt`. Większy i lepszej jakości zbiór danych zwykle jest ważniejszy niż samo zwiększanie liczby kroków.

```bat
py -m pip install torch
set PURPLE_STEPS=5000
set PURPLE_BLOCK_SIZE=512
py training/train_from_scratch.py
```

Możesz wznowić trening z istniejącego checkpointu, jeśli konfiguracja pozostaje taka sama. Aby kontynuować, ustaw `PURPLE_STEPS` na wartość większą niż liczba kroków już zapisana w checkpointcie.

Opcjonalne ustawienia: `PURPLE_BATCH_SIZE`, `PURPLE_BLOCK_SIZE`, `PURPLE_STEPS`, `PURPLE_EVAL_EVERY`, `PURPLE_LR`, `PURPLE_EMBED`, `PURPLE_HEADS`, `PURPLE_LAYERS`, `PURPLE_DROPOUT`, `PURPLE_DATA`, `PURPLE_CHECKPOINT_DIR`. `PURPLE_EMBED` musi być podzielne przez `PURPLE_HEADS`.

## Uruchomienie API (Windows)

```bat
py -m pip install -r requirements-api.txt
set PURPLE_CHECKPOINT_PATH=checkpoints\purple_ai_latest.pt
set PURPLE_API_KEY=ustaw-tu-dlugi-losowy-sekret
py -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Aby włączyć wyszukiwanie, skonfiguruj `SEARCH_API_KEY` jako sekret po stronie serwera. Nie umieszczaj klucza w aplikacji klienckiej ani w repozytorium.

## Jeden endpoint dla wszystkich klientów

`POST /v1/chat`

Nagłówek:
`Authorization: Bearer TWOJ_KLUCZ`

Przykładowe żądanie:

```json
{
  "messages": [
    {"role": "user", "content": "Cześć, przedstaw się."}
  ],
  "web_search": false,
  "max_new_tokens": 160,
  "temperature": 0.8
}
```

Włączenie `web_search: true` uruchamia wyszukiwanie po stronie API. Wyniki i adresy źródeł są przekazywane w polu `sources` odpowiedzi. To narzędzie zewnętrzne, a nie drugi model AI.

Przykładowa odpowiedź:

```json
{
  "id": "purple-chat",
  "model": "purple-ai-v1.0",
  "answer": "Tekst wygenerowany przez model...",
  "sources": [],
  "usage": {"generated_tokens": 160}
}
```

Każdy klient — CMD, strona, aplikacja mobilna, gra czy bot — wysyła takie samo żądanie HTTP. Nie potrzebuje własnego modelu.

## Pomocnicze adresy

- `GET /` — podstawowe informacje.
- `GET /health` — stan modelu i konfiguracji.
- `GET /docs` — dokumentacja techniczna API.

Jedynym publicznym endpointem rozmowy jest `POST /v1/chat`.

## Ograniczenia

To nadal mały prototyp trenowany od zera, a nie gotowy asystent na poziomie dużych modeli. Tokenizer UTF-8 rozwiązuje problem reprezentowania znaków, ale sam w sobie nie daje modelowi wiedzy ani umiejętności. Jakość zależy od danych, wielkości modelu, czasu treningu i testów. Repozytorium nie zawiera wag, a działanie API nie zostało tutaj przetestowane na wytrenowanym checkpointcie.

Przed publicznym wdrożeniem użyj HTTPS, mocnego klucza, limitów żądań i monitoringu.
