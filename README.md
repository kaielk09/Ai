# Purple AI 1.1

Projekt API **Purple AI** opartego na modelu z otwartymi wagami. Serwer ładuje model i generuje odpowiedzi samodzielnie — nie przekazuje zapytań do OpenAI.

## Co jest dostępne

- Własne API: `POST /api/v1/chat` oraz `GET /api/v1/models`.
- Podstawowa zgodność z formatem OpenAI: `POST /v1/chat/completions` oraz `GET /v1/models`.
- Opcjonalne wyszukiwanie internetowe przez Tavily.
- Ochrona kluczem API w nagłówku `Authorization: Bearer ...` lub `X-API-Key: ...`.
- Przygotowany, opcjonalny skrypt dostrajania LoRA w folderze `training/`.

## Najważniejsze pliki

- [purple_model_api.py](purple_model_api.py) — serwer modelu i oba formaty API.
- [requirements-model.txt](requirements-model.txt) — biblioteki serwera i treningu.
- [Dockerfile.model](Dockerfile.model) — obraz kontenera serwera.
- [MODEL_API.md](MODEL_API.md) — uruchomienie, konfiguracja i przykłady Python/JavaScript.
- [training/README.md](training/README.md) — jak przygotować dane i dostroić model.
- [server.js](server.js) i [public/index.html](public/index.html) — osobny, prosty interfejs demonstracyjny Node.js.

## Model i trening

Domyślnie serwer pobiera `Qwen/Qwen2.5-3B-Instruct` z Hugging Face. To gotowy model z otwartymi wagami, nie model wytrenowany od zera przez Purple AI. API udostępnia go pod nazwą `purple-1`. Folder treningowy zawiera szkielet dostrajania LoRA; rzeczywisty trening wymaga własnych danych, zgodnego środowiska i odpowiedniego sprzętu.

## Szybki start

1. Python 3.11+ i `pip install -r requirements-model.txt`.
2. Ustaw `PURPLE_API_KEY` jako długi, tajny sekret w hostingu.
3. Opcjonalnie ustaw `MODEL_ID`, `PURPLE_MODEL_NAME`, `MAX_NEW_TOKENS` i `SEARCH_API_KEY`.
4. Uruchom `uvicorn purple_model_api:app --host 0.0.0.0 --port 8000`.
5. Poczekaj na pobranie wag, sprawdź `/health` i dokumentację `/docs`.

Pełne instrukcje są w [MODEL_API.md](MODEL_API.md).

## Bezpieczeństwo i ograniczenia

Nie wkładaj prywatnego klucza API do kodu strony, aplikacji mobilnej ani publicznego repozytorium. W aplikacjach frontendowych wywołuj API przez własny backend. To nadal początkowa implementacja: nie została wdrożona ani przetestowana produkcyjnie, a streaming nie jest obsługiwany. Przed publicznym udostępnieniem dodaj limity zapytań per użytkownik, osobne klucze, monitoring, HTTPS i testy obciążeniowe. Sam GitHub/GitHub Pages nie uruchamia modelu ani API.
