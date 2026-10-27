# Purple AI — własny model i API

To repozytorium zawiera serwer, który **sam ładuje model open-weight i generuje odpowiedzi**. Nie wymaga klucza do OpenAI ani innego zewnętrznego dostawcy modelu. Wagi modelu są pobierane z Hugging Face podczas pierwszego uruchomienia — nie są przechowywane w GitHubie.

## Model domyślny

Domyślnie: `Qwen/Qwen2.5-3B-Instruct`. To gotowy model z otwartymi wagami, a nie model wytrenowany od zera przez Purple AI. Możesz zmienić go przez zmienną `MODEL_ID`, sprawdzając wcześniej licencję, wymagania sprzętowe i zgodność modelu z Transformers.

## Uruchomienie

Wymagany Python 3.11+ i wystarczająca pamięć RAM/VRAM. Model 3B może potrzebować kilku GB pamięci; CPU może działać bardzo wolno. Dla publicznego API zalecany jest serwer z odpowiednim GPU.

1. Zainstaluj zależności: `pip install -r requirements-model.txt`
2. Ustaw sekrety w środowisku hostingu:
   - `PURPLE_API_KEY` — długi, losowy sekret, który sam generujesz.
   - `MODEL_ID` — opcjonalnie, domyślnie `Qwen/Qwen2.5-3B-Instruct`.
   - `SEARCH_API_KEY` — opcjonalny klucz Tavily do wyszukiwania internetu.
   - `MAX_NEW_TOKENS` — opcjonalnie, domyślnie `768`.
3. Uruchom: `uvicorn purple_model_api:app --host 0.0.0.0 --port 8000`
4. Sprawdź `GET /health`. Pierwsze uruchomienie pobiera wagi i może trwać długo.

Możesz też zbudować kontener przez `docker build -f Dockerfile.model -t purple-ai-model .`. Uruchamiaj go z odpowiednimi zmiennymi środowiskowymi i zasobami sprzętowymi.

## Endpointy

- `GET /health` — stan serwera.
- `GET /v1/models` — lista modelu; wymaga klucza.
- `POST /v1/chat/completions` — format zgodny z podstawowym OpenAI Chat Completions API; wymaga nagłówka `Authorization: Bearer TWOJ_KLUCZ`. Streaming nie jest jeszcze obsługiwany.
- Wyszukiwanie Tavily jest opcjonalne i uruchamia się dla zapytań zawierających słowa wskazujące na aktualne informacje/wyszukiwanie.

Przykład:
```bash
curl https://TWOJ-SERWER/v1/chat/completions \
  -H "Authorization: Bearer $PURPLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "purple-1",
    "messages": [{"role": "user", "content": "Napisz funkcję sortującą w Pythonie"}],
    "max_tokens": 300
  }'
```

## Użycie z aplikacji

W SDK zgodnym z OpenAI ustaw `base_url` na `https://TWOJ-SERWER/v1`, a `api_key` na Twój klucz. Nazwa modelu to `purple-1`. W przeglądarkowym frontendzie nie umieszczaj sekretnego klucza — wywołuj API przez backend aplikacji.

## Zanim udostępnisz publicznie

To jest początkowy serwer modelu, nie kompletna komercyjna platforma. Przed otwarciem dla użytkowników dodaj limity per klucz/użytkownik, logowanie zużycia, ochronę przed nadużyciami, HTTPS, monitoring, rotację kluczy i limity kosztów hostingu. W publicznej usłudze nie udostępniaj wspólnego klucza administratora. Przetestuj wydajność i zgodność licencji wybranego modelu.

GitHub przechowuje kod. Do generowania odpowiedzi potrzebny jest działający serwer z pamięcią i mocą obliczeniową. Samo GitHub Pages nie uruchomi tego API.
