# Purple AI — trening od zera

Celem jest zbudowanie własnego małego modelu językowego: własna architektura i losowo inicjalizowane wagi, bez pobierania Qwen lub innego gotowego modelu.

## Etapy

1. Zainstaluj Python 3.11 i PyTorch zgodny ze sprzętem.
2. Przygotuj lokalny plik `training/data.txt` z tekstem, którego masz prawo używać.
3. Uruchom mały test, zanim zwiększysz rozmiar modelu lub zbioru danych.
4. Zapisuj checkpointy lokalnie i przetestuj wznowienie po przerwaniu.
5. Dopiero potem dodawaj starannie przygotowane polskie przykłady rozmów i programowania.

## Realistyczne oczekiwania

Model trenowany od zera nie zna wcześniej języka polskiego ani programowania. Musi nauczyć się wzorców z danych; mały model na domowym laptopie będzie miał ograniczone możliwości. Nie obiecujemy jakości dużych modeli komercyjnych.

## Bezpieczeństwo

- Nie zapisuj tokenów, haseł ani prywatnych rozmów w danych treningowych.
- Nie commituj `data.txt`, checkpointów ani prywatnych danych do publicznego repozytorium.
- Checkpoint to zapis stanu modelu i treningu; sam plik nie jest bazą faktów ani gwarancją poprawnych odpowiedzi.
