# Purple AI — trening od zera

Projekt zawiera własną implementację małego autoregresyjnego Transformera trenowanego od losowych wag. Nie pobiera wag Qwen ani innego gotowego modelu.

## Tokenizer

Model używa tokenizera bajtowego UTF-8. Każdy bajt ma identyfikator 0–255, więc tekst może zawierać polskie znaki i inne znaki Unicode. To prosty tokenizer bazowy, a nie zaawansowany tokenizer subword.

## Uruchomienie

1. Zainstaluj Python 3.11.
2. Zainstaluj PyTorch odpowiedni dla komputera: https://pytorch.org/get-started/locally/
3. Skopiuj `training/data.example.txt` do `training/data.txt`.
4. Dodaj legalnie używalne dane tekstowe. Najlepiej przygotować przykłady w formacie rozmów, pytania i odpowiedzi, wyjaśnienia, dokumentację oraz kod z odpowiednimi licencjami.
5. Uruchom z katalogu głównego:

```bat
py training/train_from_scratch.py
```

Domyślnie skrypt wykonuje 5000 kroków z kontekstem 512 bajtów. Checkpoint zapisuje do `checkpoints/purple_ai_latest.pt`.

## Ustawienia środowiskowe

- `PURPLE_STEPS` — łączna liczba kroków treningu (domyślnie 5000).
- `PURPLE_BATCH_SIZE` — liczba przykładów w partii (16).
- `PURPLE_BLOCK_SIZE` — długość kontekstu w bajtach (512).
- `PURPLE_EVAL_EVERY` — częstotliwość ewaluacji (100).
- `PURPLE_LR` — learning rate (0.0003).
- `PURPLE_EMBED` — szerokość reprezentacji (192).
- `PURPLE_HEADS` — liczba głów uwagi (4).
- `PURPLE_LAYERS` — liczba bloków (4).
- `PURPLE_DROPOUT` — dropout (0.1).
- `PURPLE_DATA` — ścieżka do pliku treningowego.
- `PURPLE_CHECKPOINT_DIR` — katalog checkpointów.

`PURPLE_EMBED` musi być podzielne przez `PURPLE_HEADS`. Aby wznowić trening, użyj tego samego zbioru i ustawień oraz ustaw `PURPLE_STEPS` wyżej niż krok zapisany w checkpointcie.

## Ważne

- Nie ma gotowych wag w repozytorium. Trzeba je samodzielnie wytrenować.
- Mały model i mały zbiór danych nie dadzą jakości dużego asystenta.
- Tokenizer UTF-8 pozwala reprezentować znaki; nie daje sam z siebie wiedzy.
- Przed użyciem danych sprawdź ich licencje. Nie dodawaj haseł, kluczy, danych prywatnych ani prywatnych rozmów.
- Nie commituj `training/data.txt` ani folderu `checkpoints/`.
