# Purple AI — trening od zera

Celem jest zbudowanie własnego małego modelu językowego: własna architektura i losowo inicjalizowane wagi, bez pobierania Qwen lub innego gotowego modelu.

## Uruchomienie prototypu

1. Zainstaluj Python 3.11.
2. Zainstaluj PyTorch odpowiedni dla komputera. Instrukcje: https://pytorch.org/get-started/locally/
3. Skopiuj `training/data.example.txt` do `training/data.txt`.
4. Dodaj do lokalnego `training/data.txt` tekst, do którego masz prawo. Możesz użyć przykładów polskiego, pytań i odpowiedzi oraz kodu.
5. Uruchom z katalogu głównego repozytorium:

```bash
python training/train_from_scratch.py
```

Skrypt zapisuje `checkpoints/purple_ai_latest.pt` oraz słownik znaków. Gdy uruchomisz go ponownie z tym samym plikiem danych i ustawieniami, spróbuje wznowić trening z checkpointu.

## Opcjonalne ustawienia

Zmienne środowiskowe:
- `PURPLE_STEPS` — liczba kroków treningu (domyślnie 2000).
- `PURPLE_BATCH_SIZE` — rozmiar partii (domyślnie 16).
- `PURPLE_BLOCK_SIZE` — długość kontekstu znaków (domyślnie 128).
- `PURPLE_CHECKPOINT_DIR` — folder checkpointów.

## Realistyczne oczekiwania

To mały prototyp znakowego modelu autoregresyjnego. Nie jest jeszcze praktycznym chatbotem i nie będzie od razu dobrze odpowiadać na pytania ani pisać programów. Najpierw sprawdzimy, czy trening, zapisywanie i wznawianie działają; później będziemy rozwijać architekturę i przygotowywać większe, starannie sprawdzone dane.

## Bezpieczeństwo

- Nie zapisuj tokenów, haseł ani prywatnych rozmów w danych treningowych.
- Nie commituj `training/data.txt` ani folderu `checkpoints/` do repozytorium.
