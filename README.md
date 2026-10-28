# Purple AI V1.0 — model trenowany od zera

Purple AI jest przebudowywany jako **mały model językowy trenowany od zera**. Nie używamy Qwen ani innego gotowego modelu bazowego.

## Ważne ograniczenia

- Własna architektura i własne wagi nie oznaczają, że model od razu będzie umiał dobrze rozmawiać lub programować.
- Mały model edukacyjny wymaga dobrych danych i wielu testów; na laptopie z RTX 2060 6 GB trzeba zacząć od niewielkich ustawień.
- GitHub przechowuje kod, a nie uruchamia treningu. Trening odbędzie się lokalnie na komputerze.
- Checkpointy będą zapisywać stan modelu, optymalizatora i numer kroku, aby można było próbować wznowić trening.
- Nie dodawaj kluczy API, prywatnych danych ani checkpointów do publicznego repozytorium.

## Struktura projektu

- `training/train_from_scratch.py` — punkt startowy dla własnego małego modelu.
- `training/data.txt` — własny tekst treningowy (tworzysz lokalnie; nie commituj prywatnych danych).
- `checkpoints/` — lokalne zapisane stany modelu; nie wrzucamy ich do GitHuba.
- `server.js` i `public/index.html` — istniejący interfejs demonstracyjny; nie są samym modelem AI.

## Następny etap

Po przygotowaniu Pythona i środowiska na komputerze uruchomimy mały trening testowy, sprawdzimy zapis checkpointów, a następnie stopniowo przygotujemy dane polskie, odpowiedzi i przykłady programowania.

Projekt jest w trakcie przebudowy. Nie ma jeszcze wytrenowanego modelu Purple AI.
