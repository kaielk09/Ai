# Purple AI V1.0 — jedno API rozmowy

Wszystkie klienty korzystają z tego samego endpointu: `POST /v1/chat`.
Może to być program CMD/PowerShell, aplikacja desktopowa lub mobilna, strona, gra albo bot Discord. Klient wysyła wiadomości do API; nie uruchamia własnej kopii modelu.

## Ważne: potrzebny jest checkpoint

Repozytorium zawiera kod modelu, ale nie zawiera wytrenowanych wag. API nie będzie udawać, że model jest gotowy, jeśli checkpoint nie istnieje.

Domyślna ścieżka:
`checkpoints/purple_ai_latest.pt`

## Uruchomienie w Windows

W katalogu głównym repozytorium:

```bat
py -m pip install -r requirements-api.txt
set PURPLE_CHECKPOINT_PATH=checkpoints\purple_ai_latest.pt
set PURPLE_API_KEY=ustaw-tu-dlugi-losowy-sekret
py -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Jeśli chcesz włączyć wyszukiwanie internetu, ustaw po stronie serwera sekret `SEARCH_API_KEY`. Klucz Tavily nigdy nie powinien być umieszczany w aplikacji klienta ani w publicznym repozytorium.

## Jedyny endpoint rozmowy

`POST /v1/chat`

Nagłówek:
`Authorization: Bearer TWOJ_KLUCZ`

Przykładowe JSON:

```json
{
  "messages": [
    {"role": "user", "content": "Cześć, przedstaw się."}
  ],
  "web_search": false,
  "max_new_chars": 300,
  "temperature": 0.8
}
```

Aby poprosić o wyszukanie informacji, ustaw `web_search` na `true`. API samo uruchomi wyszukiwanie, przekaże materiały do kontekstu modelu i zwróci znalezione źródła w polu `sources`. Wyszukiwanie wymaga skonfigurowanego `SEARCH_API_KEY`.

Przykładowa odpowiedź:

```json
{
  "id": "purple-chat",
  "model": "purple-ai-v1.0",
  "answer": "Tekst wygenerowany przez model...",
  "sources": [],
  "usage": {"generated_characters": 25}
}
```

## Podłączenie dowolnej aplikacji

Każdy klient wykonuje żądanie HTTP POST do tego samego adresu `/v1/chat`, wysyłając historię rozmowy i klucz API. Język programowania klienta nie ma znaczenia. Nie trzeba tworzyć osobnego modelu dla gry, strony, CMD czy Discorda.

## Diagnostyka

- `GET /` — podstawowe informacje.
- `GET /health` — stan modelu i konfiguracji serwera.
- `GET /docs` — automatyczna dokumentacja techniczna FastAPI.

Te adresy są pomocnicze; jedynym publicznym endpointem rozmowy jest `POST /v1/chat`.

## Ograniczenia obecnej wersji

Aktualny model to mały znakowy Transformer trenowany od losowych wag. Nie jest jeszcze zaawansowanym asystentem: nie ma tokenizera subword, a jego jakość zależy od danych i treningu. Wyszukiwarka dostarcza źródła, ale sam model może nie umieć ich poprawnie podsumować. Nie ma jeszcze podstaw, by nazywać go ukończonym lub porównywać z dużymi modelami komercyjnymi.

Przed publicznym wdrożeniem potrzebne są HTTPS, silny sekret API, limity żądań, monitoring i testy obciążeniowe.
