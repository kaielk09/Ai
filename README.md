# Purple AI 1.0

Projekt własnego API AI opartego na modelu z otwartymi wagami. Serwer ładuje model i generuje odpowiedzi samodzielnie — nie przekazuje zapytań do API OpenAI.

## Najważniejsze pliki

- [purple_model_api.py](purple_model_api.py) — serwer modelu oraz endpointy API zgodne z podstawowym formatem OpenAI Chat Completions.
- [requirements-model.txt](requirements-model.txt) — biblioteki Pythona.
- [Dockerfile.model](Dockerfile.model) — obraz kontenera serwera.
- [MODEL_API.md](MODEL_API.md) — instrukcja uruchomienia, konfiguracji i korzystania z API.
- [server.js](server.js) i [public/index.html](public/index.html) — osobny, prosty interfejs demonstracyjny Node.js.

## Jak działa własny model?

Domyślnie serwer pobiera model `Qwen/Qwen2.5-3B-Instruct` z Hugging Face przy pierwszym uruchomieniu. To gotowy model z otwartymi wagami, nie model wytrenowany od zera przez Purple AI. Wagi nie są umieszczone w GitHubie. Do uruchomienia wymagany jest serwer z odpowiednią pamięcią RAM/VRAM; na CPU generowanie może być bardzo wolne.

## Szybki start

1. Użyj Pythona 3.11 lub nowszego i zainstaluj zależności: `pip install -r requirements-model.txt`.
2. Ustaw w zmiennych środowiskowych hostingu `PURPLE_API_KEY` jako własny, długi sekret. Opcjonalnie ustaw `MODEL_ID` i `SEARCH_API_KEY`.
3. Uruchom `uvicorn purple_model_api:app --host 0.0.0.0 --port 8000`.
4. Poczekaj na pobranie i załadowanie modelu, a następnie sprawdź `/health`.

Pełne instrukcje i przykładowe zapytanie znajdziesz w [MODEL_API.md](MODEL_API.md).

## Podłączenie w aplikacji

Ustaw adres bazowy SDK zgodnego z OpenAI na `https://TWOJ-SERWER/v1`, klucz na wartość `PURPLE_API_KEY`, a nazwę modelu na `purple-1`. Przykład:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://TWOJ-SERWER/v1",
    api_key="TWOJ_KLUCZ_PURPLE",
)

response = client.chat.completions.create(
    model="purple-1",
    messages=[{"role": "user", "content": "Cześć, Purple AI!"}],
)
print(response.choices[0].message.content)
```

Nie wkładaj prywatnego klucza API do kodu strony ani publicznego repozytorium. Dla aplikacji webowych wywołuj API z backendu.

## Ważne ograniczenia

To początkowa implementacja modelu/API, nie gotowa komercyjna platforma. Nie została jeszcze wdrożona na publiczny serwer ani przetestowana produkcyjnie. Przed udostępnieniem innym użytkownikom potrzebne są testy, limity zapytań, zarządzanie osobnymi kluczami, monitoring i zabezpieczenia przed nadużyciami. Sam GitHub/GitHub Pages nie uruchamia modelu ani API.