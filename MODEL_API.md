# Purple AI — API i przygotowanie treningu

Purple AI ma teraz dwa interfejsy:
- **Własne API Purple:** `POST /api/v1/chat` i `GET /api/v1/models`
- **Zgodność z OpenAI:** `POST /v1/chat/completions` i `GET /v1/models`, dzięki czemu łatwo podłączyć biblioteki i narzędzia obsługujące ten format.

## Model
Domyślny model bazowy to `Qwen/Qwen2.5-3B-Instruct`, wystawiany w API pod nazwą `purple ai V1.0`. To gotowy model open-weight, nie model trenowany od zera przez Purple AI. Kod dodaje warstwę API i opcjonalne wyszukiwanie; nie sprawia automatycznie, że model staje się „100 razy mądrzejszy”.

## Uruchomienie
Python 3.11+ i odpowiednia pamięć RAM/VRAM są wymagane. Pierwszy start pobiera wagi z Hugging Face.

1. `pip install -r requirements-model.txt`
2. Ustaw zmienne środowiskowe w hostingu:
   - `PURPLE_API_KEY` — długi, tajny klucz.
   - `MODEL_ID` — opcjonalnie, domyślnie `Qwen/Qwen2.5-3B-Instruct`.
   - `PURPLE_MODEL_NAME` — opcjonalnie, domyślnie `purple ai V1.0`.
   - `SEARCH_API_KEY` — opcjonalny klucz Tavily do wyszukiwania.
   - `MAX_NEW_TOKENS` — limit generowania, domyślnie 768.
3. Uruchom: `uvicorn purple_model_api:app --host 0.0.0.0 --port 8000`
4. Dokumentacja interaktywna: `/docs`; stan: `/health`.

## Autoryzacja
Każde żądanie API (poza publicznym `/health` i stroną główną) wymaga jednego z nagłówków:
`Authorization: Bearer TWÓJ_KLUCZ` lub `X-API-Key: TWÓJ_KLUCZ`.

Nigdy nie wkładaj klucza serwerowego do publicznego frontendu, aplikacji mobilnej ani repozytorium. Aplikacje klienckie powinny wywoływać własny backend, który przechowuje sekret.

## Własne API Purple AI
Przykład Python:
```python
import requests

BASE_URL = "https://TWOJ-SERWER"
API_KEY = "TWÓJ_KLUCZ"
response = requests.post(
    BASE_URL + "/api/v1/chat",
    headers={"X-API-Key": API_KEY},
    json={
        "messages": [{"role": "user", "content": "Napisz funkcję sortującą w Pythonie"}],
        "temperature": 0.4,
        "max_tokens": 400,
        "web_search": False
    },
    timeout=120
)
response.raise_for_status()
print(response.json()["output"])
```

Przykład JavaScript (uruchamiaj po stronie backendu, nie w publicznej przeglądarce):
```javascript
const response = await fetch("https://TWOJ-SERWER/api/v1/chat", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.PURPLE_API_KEY
  },
  body: JSON.stringify({
    messages: [{ role: "user", content: "Witaj, Purple AI!" }],
    max_tokens: 300
  })
});
if (!response.ok) throw new Error(`Purple AI API error: ${response.status}`);
const data = await response.json();
console.log(data.output);
```

Własna odpowiedź API zawiera `output`, `model`, `usage` i `sources`. Aby wymusić wyszukiwanie, ustaw `web_search: true`; działa tylko po ustawieniu `SEARCH_API_KEY`.

## Zgodność z OpenAI
Ustaw base URL na `https://TWOJ-SERWER/v1`, klucz na swój klucz Purple AI, a nazwę modelu na `purple ai V1.0`. To zgodność na poziomie podstawowego formatu czatu, nie pełna implementacja każdego endpointu OpenAI. Streaming nie jest obecnie obsługiwany.

## Trening / dostrajanie
W folderze `training/` są skrypt LoRA i przykład formatu danych. To **dostrajanie istniejącego modelu**, nie trening od zera. Potrzebujesz własnych, legalnie pozyskanych przykładów wysokiej jakości i sprzętu obsługującego trening. Najpierw skopiuj `training/data.example.jsonl` do `training/data.jsonl`, zastąp przykłady własnymi i uruchom instrukcje z `training/README.md`. Nie commituj prywatnych danych, sekretów ani danych osobowych. Skrypt nie został uruchomiony na sprzęcie treningowym w ramach tej zmiany.

## Ważne ograniczenia
- Sam GitHub nie hostuje uruchomionego modelu; potrzebny jest osobny serwer/hosting.
- Przed publicznym uruchomieniem dodaj limity użycia per użytkownik/klucz, monitoring, ochronę przed nadużyciami, HTTPS, rotację kluczy i testy obciążeniowe.
- `/health` pokazuje nazwę typu błędu ładowania, ale nie ujawnia sekretów ani pełnego wyjątku.
- Wyszukiwanie internetowe jest opcjonalne i wymaga klucza Tavily.
