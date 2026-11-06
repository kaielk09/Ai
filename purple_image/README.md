# Purple AI Image — autorski generator od zera

To osobny projekt generatora obrazow w repozytorium Purple AI. Architektura sieci, enkoder bajtow tekstu, warunkowanie, petla treningowa i sampler sa napisane w tym repozytorium. Nie ma tu gotowych wag, gotowego generatora, Diffusers, Transformers ani zewnetrznego API obrazowego.

## Dozwolone narzedzia

- PyTorch: tensory, automatyczne rozniczkowanie, optymalizator i obliczenia GPU.
- Pillow: wczytywanie i zapisywanie plikow obrazow (wylacznie I/O).
- NumPy: konwersja pikseli (wylacznie I/O).

Nie korzystamy z gotowych generatorow, modeli bazowych ani gotowych wag. Wlasna architektura jest zdefiniowana w `model.py`.

## Pliki

- `model.py` — autorski enkoder UTF-8, embedding czasu, bloki rezydualne i siec przewidujaca szum.
- `train.py` — trening od losowej inicjalizacji na parach obraz/opis.
- `generate.py` — lokalne generowanie PNG z checkpointu.
- `data/` — dane treningowe.
- `checkpoints/` — lokalne wagi modelu.

## Przygotowanie danych

Utworz katalog `data/images/` i plik `data/captions.jsonl`. Kazda linia ma format:

```json
{"image":"images/obraz_001.jpg","caption":"Fioletowy zamek na wzgorzu o zachodzie slonca"}
{"image":"images/obraz_002.png","caption":"Czerwony samochod na tle gor"}
```

Trenuj tylko na obrazach, do ktorych masz prawa i ktorych licencja pozwala na trening. Opisy powinny wiernie opisywac obrazy.

## Instalacja

Python 3.10+:

```bash
python -m pip install -r requirements.txt
```

## Trening — eksperymentalna wersja 64x64

```bash
python train.py --data data --epochs 10 --batch-size 1 --size 64
```

## Generowanie

```bash
python generate.py --prompt "Fioletowy smok lecacy nad zamkiem" --output generated/test.png
```

## Ograniczenia

To prototyp badawczy, nie generator wysokiej jakosci. Nie ma jeszcze wytrenowanych wag. Wyniki na poczatku beda slabe; trening moze wymagac wielu godzin lub dni. RTX 2060 6 GB ogranicza rozmiar modelu, rozdzielczosc i batch. Nie obiecujemy jakosci ChatGPT ani profesjonalnych generatorow. Najpierw sprawdzamy, czy pipeline uczy sie na malym, legalnym zbiorze, potem rozwijamy architekture i jakosc.

Checkpointy i dane nie sa automatycznie publikowane na GitHubie.
