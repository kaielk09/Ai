# Purple AI V1.0 — model trenowany od zera

Purple AI jest przebudowywany jako **mały model językowy trenowany od zera**. Nie używamy Qwen ani innego gotowego modelu bazowego.

## Co jest teraz w repozytorium

- `training/train_from_scratch.py` — prosty prototyp małego modelu znakowego z własnymi, losowo inicjalizowanymi wagami.
- `training/data.example.txt` — mały przykład tekstu treningowego.
- `training/README.md` — plan kolejnych kroków.
- `server.js` i `public/index.html` — dotychczasowy interfejs demonstracyjny; nie są samym modelem AI.

## Jak uruchomić prototyp później

1. Zainstaluj Python 3.11.
2. Zainstaluj PyTorch zgodny z komputerem (na początek wersja CPU też pozwoli sprawdzić kod).
3. Skopiuj `training/data.example.txt` do `training/data.txt`.
4. Uruchom `python training/train_from_scratch.py`.
5. Checkpointy będą zapisywane lokalnie w folderze `checkpoints/`.

## Ważne ograniczenia

- To mały **prototyp edukacyjny**, a nie gotowy asystent. Uczy się przewidywać następny znak; na małej próbce nie będzie dobrze odpowiadał na dowolne pytania ani dobrze programował.
- Następnie rozbudujemy architekturę, dane i testy, aby pracować w stronę modelu rozmownego.
- Checkpointy zapisują stan modelu i optymalizatora, ale wznowienie wymaga tych samych danych i ustawień.
- GitHub przechowuje kod, ale sam nie uruchamia treningu. Trening będzie wykonywany lokalnie.
- `.gitignore` wyklucza lokalne dane i checkpointy. Nie wrzucaj sekretów, prywatnych danych ani wytrenowanych wag do publicznego repozytorium.

Projekt jest w trakcie przebudowy. Nie ma jeszcze wytrenowanego modelu Purple AI.
