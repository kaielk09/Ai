# Purple AI Image — wspólny model tekstowo-obrazowy

## Architektura

Generator obrazów jest częścią jednego systemu Purple AI i używa tego samego rdzenia tekstowego co czat i programowanie.

- `api/app.py`: istniejący model tekstowy ma metodę `encode()`, która zwraca kontekstowe reprezentacje promptu.
- `purple_image/model.py`: autorska sieć obrazowa przyjmuje te reprezentacje bez własnego tokenizera i enkodera tekstu.
- `purple_image/train.py`: ładuje checkpoint tekstowy i zamraża jego wagi; uczy tylko nową część obrazową.
- `purple_image/generate.py`: sampler używany przez istniejący endpoint API.

Nie ma zewnętrznego generatora, obcego modelu bazowego ani osobnego API obrazowego.

## Dlaczego tekstowy model jest zamrożony?

To chroni istniejące umiejętności rozmowy, pisania i programowania. Model tekstowy jest współdzielony i dostarcza rozumienie opisu, natomiast trening obrazu uczy nową część odpowiedzialną za tworzenie pikseli. Nie należy automatycznie odmrażać całego rdzenia tekstowego, bo trening na małym zbiorze obrazów mógłby pogorszyć czat i kodowanie.

## Trening

Wymagany jest istniejący checkpoint tekstowy Purple AI oraz legalny zbiór par obraz-opis w `purple_image/data/captions.jsonl`. Szczegóły formatu i polecenia są w `purple_image/README.md`.

CPU jest obsługiwany, ale trening generatora od zera na CPU jest bardzo wolny. Nie istnieje uczciwy sposób, by obiecać wysoką jakość po krótkim treningu CPU. Pierwsze uruchomienie jest testem potoku, nie gotowym modelem.

## Zasady

- Nie usuwaj ani nie nadpisuj checkpointu tekstowego.
- Nie pobieraj gotowych wag innych generatorów.
- Nie wysyłaj promptów ani obrazów do zewnętrznego dostawcy.
- Nie ogłaszaj generatora jako gotowego przed treningiem, testem zgodności wymiarów i oceną próbek.
- Jeśli nowy checkpoint obrazowy nie istnieje, API ma zwrócić czytelny błąd.
