# Purple AI Image — generator z jednym wspólnym rdzeniem tekstowym

Moduł obrazowy używa **tego samego modelu tekstowego Purple AI**, który obsługuje rozmowy i programowanie. API pobiera ukryte reprezentacje z `PurpleGPT.encode()` i przekazuje je do autorskiej sieci dyfuzyjnej. W `purple_image/model.py` nie ma drugiego tokenizera ani drugiego enkodera tekstu.

## Co jest autorskie

- Własna sieć obrazu U-Net z blokami rezydualnymi i cross-attention.
- Własny sampler dyfuzyjny.
- Warunkowanie obrazów ukrytymi reprezentacjami istniejącego modelu Purple AI.
- Brak gotowych wag generatorów, zewnętrznych usług obrazowych i modeli bazowych.
- PyTorch służy do tensorów, gradientów i obliczeń.

## Ochrona istniejących umiejętności

Trening obrazu ładuje aktualny checkpoint tekstowy wskazany przez `PURPLE_CHECKPOINT_PATH` i **zamraża jego wagi**. Uczy się wyłącznie nowa część obrazowa. Dzięki temu trening obrazów nie nadpisuje wiedzy tekstowej ani programistycznej. Wagi tekstowe nie są kopiowane do checkpointu obrazu — API używa tego samego załadowanego modelu.

## Dane treningowe

Przygotuj katalog:

```text
purple_image/data/
  captions.jsonl
  images/
    image_001.jpg
    image_002.png
```

Każdy wiersz `captions.jsonl` to osobna para:

```json
{"image":"images/image_001.jpg","caption":"Fioletowy zamek na wzgórzu o zachodzie słońca"}
{"image":"images/image_002.png","caption":"Czerwony samochód na tle gór"}
```

Używaj wyłącznie obrazów, do których masz prawa i których licencja pozwala na trening. Opisy muszą wiernie opisywać obraz. Nie ma automatycznego pobierania przypadkowych zdjęć z internetu.

## Trening

Zainstaluj zależności repozytorium i ustaw ścieżkę do istniejącego checkpointu tekstowego:

```bash
export PURPLE_CHECKPOINT_PATH=/sciezka/do/purple_ai_latest.pt
python -m pip install -r purple_image/requirements.txt
python -m purple_image.train --data purple_image/data --size 64 --epochs 1
```

Windows PowerShell:

```powershell
$env:PURPLE_CHECKPOINT_PATH="C:\sciezka\do\purple_ai_latest.pt"
python -m pip install -r purple_image/requirements.txt
python -m purple_image.train --data purple_image/data --size 64 --epochs 1
```

Wznowienie:

```bash
python -m purple_image.train --data purple_image/data --size 64 --epochs 5 --resume
```

`--epochs` oznacza docelową liczbę epok. Trening zapisuje checkpoint po każdej epoce do `checkpoints/purple_image.pt`.

## API

Endpoint pozostaje ten sam: `POST /v1/images/generations`. Wymaga tego samego klucza API co czat oraz checkpointu `checkpoints/purple_image.pt` w formacie `purple-shared-core-v2`. Jeśli nowy checkpoint nie istnieje, API zwraca błąd zamiast udawać, że wygenerowało obraz.

API bierze prompt przez istniejący tokenizer tekstowy Purple AI, wylicza reprezentacje `_model.encode(...)` i przekazuje je do generatora. Wymiar reprezentacji jest sprawdzany przy generowaniu.

## Ograniczenia

- To autorski eksperymentalny generator trenowany od losowej inicjalizacji, nie gotowy generator komercyjnej jakości.
- CPU pozwala uruchomić testy i mały trening, ale trening wysokiej jakości od zera będzie skrajnie wolny. Wysoka jakość wymaga dużego, różnorodnego zbioru par obraz-opis i długiego treningu na GPU.
- Trening na 64x64 nie nauczy modelu szczegółów wysokiej rozdzielczości. Samo ustawienie większego rozmiaru podczas generowania nie tworzy nowych szczegółów.
- Pierwszy trening na CPU należy traktować jako test działania, nie jako gotowy model.
