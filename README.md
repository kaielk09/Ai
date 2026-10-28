# Purple AI V1.0 — model trenowany od zera

Purple AI jest rozwijany jako **mały model językowy trenowany od zera**. Nie używa Qwen ani innego gotowego modelu bazowego.

## Pliki projektu

- `training/train_from_scratch.py` — prototyp znakowego Transformera z losowo inicjalizowanymi wagami.
- `training/data.example.txt` — mały przykład tekstu treningowego.
- `api/app.py` — API HTTP do uruchamiania wytrenowanego checkpointu.
- `api/README.md` — instrukcja instalacji i testowania API.
- `requirements-api.txt` — zależności API.
- `server.js` i `public/index.html` — dotychczasowy interfejs demonstracyjny; nie są samym modelem AI.

## Trening lokalny

1. Zainstaluj Python 3.11 i PyTorch zgodny z komputerem.
2. Skopiuj `training/data.example.txt` do `training/data.txt` i dodaj legalne, dobrej jakości przykłady.
3. Uruchom z katalogu głównego repozytorium: `python training/train_from_scratch.py`.
4. Checkpoint jest zapisywany lokalnie do `checkpoints/purple_ai_latest.pt`.

## Uruchomienie API (Windows CMD)

Po utworzeniu checkpointu z treningu, z katalogu głównego repozytorium uruchom:

```bat
py -m pip install -r requirements-api.txt
set PURPLE_CHECKPOINT_PATH=checkpoints\purple_ai_latest.pt
set PURPLE_API_KEY=ustaw-tu-wlasny-dlugi-sekret
py -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Dokumentacja interaktywna: `http://127.0.0.1:8000/docs`; status modelu: `http://127.0.0.1:8000/health`. Szczegóły żądań są w `api/README.md`. API działa na komputerze, na którym je uruchomisz — dodanie plików na GitHubie samo go nie hostuje.

## Ograniczenia — przeczytaj

- Repozytorium ma kod, ale **nie zawiera jeszcze wytrenowanych wag**. API nie wygeneruje tekstu, dopóki nie będzie dostępny checkpoint zgodny z kodem.
- To prototyp znakowego modelu przewidującego kolejny znak, a nie gotowy asystent. Nie będzie jeszcze wiarygodnie odpowiadał na dowolne pytania.
- Wznowienie treningu wymaga tych samych danych i ustawień.
- GitHub przechowuje kod, ale sam nie uruchamia treningu.
- Nie commituj tokenów, prywatnych danych treningowych ani checkpointów do publicznego repozytorium.
