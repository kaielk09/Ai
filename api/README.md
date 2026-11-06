# Purple AI — jedno wspólne API

Purple AI udostępnia czat tekstowy i generowanie obrazów w tej samej aplikacji FastAPI. Oba używają jednego adresu serwera i wspólnego nagłówka Authorization: Bearer PURPLE_API_KEY.

## Endpointy
- POST /v1/chat — czat Purple AI.
- POST /v1/chat/completions — format OpenAI Chat Completions.
- POST /v1/images/generations — Purple AI Image.
- GET /v1/models — lista obu modeli.
- GET /health — status modelu tekstowego.
- GET /docs — dokumentacja API.

## Uruchomienie
```bash
python -m pip install -r requirements-api.txt
export PURPLE_API_KEY='ustaw-wlasny-sekret'
export PURPLE_IMAGE_DEVICE=cpu
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Generowanie obrazu
```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H 'Authorization: Bearer TWOJ_KLUCZ' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ciemny fioletowy smok z jasnymi oczami","width":124,"height":124,"steps":30,"seed":321}'
```

Odpowiedź zawiera PNG w Base64 w polu data[0].b64_json, rozmiar oraz paletę kolorów.

## Rozmiary
Żądanie przyjmuje niezależne width i height. Każdy bok może mieć od 16 do 1024 pikseli, a maksymalny obszar wynosi 1 048 576 pikseli. Przykłady: 124x124, 512x512, 1024x1024. Większe proporcje są ograniczone limitem łącznej liczby pikseli.

## Kolory
purple_image/colors.py rozpoznaje popularne nazwy polskie i angielskie, dowolne kolory HEX oraz RGB. Tworzy paletę harmonijnych odcieni i dopisuje wskazówki kolorystyczne do promptu modelu. To pomoc regułowa; dokładne odwzorowanie kolorów wymaga odpowiednich przykładów w danych treningowych.

## Ważne ograniczenia
Repozytorium nie zawiera wytrenowanego checkpointu obrazu. Bez treningu losowe wagi nie potrafią tworzyć semantycznie poprawnych obrazów. Obsługa większego rozmiaru oznacza, że sieć może pracować na takim tensorze, a nie że nauczyła się szczegółów w tej rozdzielczości. Używaj checkpointu trenowanego na odpowiedniej rozdzielczości. Colab jest środowiskiem testowym, a nie stałym hostingiem API.

Nie umieszczaj klucza API w publicznym kodzie klienta. W produkcji używaj HTTPS i przechowuj sekret wyłącznie po stronie serwera.