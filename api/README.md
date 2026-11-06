# Purple AI V1.0 — wspólne API dla wszystkich urządzeń

Wszystkie klienty korzystają z jednego serwera i tych samych wag modelu: Android, iPhone, przeglądarka, Windows, CMD, bot Discord i inne aplikacje. Nie trzeba trenować osobnego modelu dla każdego urządzenia.

## Endpointy

- `POST /v1/chat` — natywny format Purple AI (z opcjonalnym wyszukiwaniem).
- `POST /v1/chat/completions` — format zgodny z OpenAI Chat Completions, bez streamingu.
- `GET /v1/models` — lista modeli.
- `GET /health` — stan modelu.
- `GET /` — informacje o API.
- `GET /docs` — dokumentacja Swagger.

## Uruchomienie API (Windows)

Z głównego katalogu repozytorium:

```bat
py -m pip install -r requirements-api.txt
set PURPLE_CHECKPOINT_PATH=checkpoints\\purple_ai_latest.pt
set PURPLE_API_KEY=ustaw-tu-dlugi-losowy-sekret
py -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

W publicznym wdrożeniu ustaw HTTPS i przechowuj sekrety wyłącznie po stronie serwera.

## Konfiguracja CORS

Dla klientów API ma CORS. Zmienna `PURPLE_CORS_ORIGINS` przyjmuje listę originów oddzielonych przecinkami, np.:

```text
PURPLE_CORS_ORIGINS=https://twojastrona.pl,https://app.twojastrona.pl
```

Domyślnie ustawione jest `*` dla łatwiejszych testów. W produkcji ustaw własną listę domen. CORS nie zastępuje uwierzytelniania.

## Przykład: OpenAI-compatible

```bash
curl https://TWOJ-SERWER/v1/chat/completions \
  -H "Authorization: Bearer TWOJ_KLUCZ_API" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "purple-ai-v1.0",
    "messages": [{"role": "user", "content": "Cześć! Co potrafisz?"}],
    "max_tokens": 160,
    "temperature": 0.8
  }'
```

Tekst odpowiedzi znajduje się w `choices[0].message.content`.

## Przykład: natywny format Purple AI

```bash
curl https://TWOJ-SERWER/v1/chat \
  -H "Authorization: Bearer TWOJ_KLUCZ_API" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Cześć!"}],
    "web_search": false,
    "max_new_tokens": 160,
    "temperature": 0.8,
    "top_k": 40
  }'
```

## Bezpieczeństwo

**Nie wkładaj `PURPLE_API_KEY` do kodu aplikacji Android/iOS, bota ani programu klienckiego.** Każdy użytkownik mógłby go odczytać. Dla publicznych aplikacji użyj własnego backendu/proxy, który przechowuje klucz i przekazuje żądania do Purple AI. Używaj HTTPS.

Opcjonalne wyszukiwanie wymaga `SEARCH_API_KEY` ustawionego po stronie serwera i jest dostępne przez natywny endpoint `/v1/chat`.

## Model i ograniczenia

API obsługuje tokenizer bajtowy UTF-8 oraz kompatybilne checkpointy znakowe, jeśli zawierają oryginalną mapę tokenów. API nie zgaduje brakującej mapy. Zgodność formatu API nie oznacza jakości porównywalnej z dużymi modelami komercyjnymi. Repozytorium samo w sobie nie dostarcza hostingu ani wytrenowanych wag — serwer musi mieć checkpoint, zależności i zasoby do uruchomienia modelu.
