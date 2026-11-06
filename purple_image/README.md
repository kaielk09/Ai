# Purple AI Image — autorski generator od zera

Oddzielny projekt text-to-image Purple AI. Architektura sieci, enkoder tekstu UTF-8, uwaga tekst-przestrzen, siec U-Net, trening dyfuzyjny i sampler sa napisane w tym repozytorium. Nie ma gotowych wag, modelu bazowego, Diffusers, Transformers ani zewnetrznego API obrazowego.

## Zasady techniczne

- **PyTorch**: tylko tensory, autograd, optymalizator i obliczenia GPU.
- **Pillow**: tylko odczyt i zapis plikow obrazow.
- **NumPy**: konwersja pikseli.
- Nie pobieramy gotowych generatorow, modeli bazowych ani cudzych wag.
- Wlasna architektura znajduje sie w `model.py`.

## Pliki

- `model.py` — enkoder bajtow UTF-8 z autorska uwaga, bloki rezydualne, uwaga tekstu na mapie cech i siec U-Net przewidujaca szum.
- `train.py` — trening od losowej inicjalizacji, podzial walidacyjny, EMA wag, AMP na CUDA, zapisywanie checkpointow i wznawianie.
- `generate.py` — lokalny sampler DDIM i generowanie PNG z wag EMA.
- `check_data.py` — sprawdzanie pliku opisow i poprawnosci obrazow.
- `test_model.py` — test ksztaltu wyjscia i przeplywu gradientow bez datasetu.
- `requirements.txt` — minimalne zaleznosci.

## Przygotowanie danych

Utworz strukture:

```text
purple_image/
  data/
    captions.jsonl
    images/
      image_001.jpg
      image_002.png
```

Przykladowy `data/captions.jsonl`:

```json
{"image":"images/image_001.jpg","caption":"Fioletowy zamek na wzgorzu o zachodzie slonca"}
{"image":"images/image_002.png","caption":"Czerwony samochod na tle gor"}
```

Uzywaj wylacznie obrazow, do ktorych masz prawa i ktorych warunki pozwalaja na trening. Opisy powinny wiernie opisywac obrazy. Nie wrzucaj prywatnych ani wrazliwych zdjec.

## Instalacja

Python 3.10+:

```bash
python -m pip install -r requirements.txt
```

## Kontrola danych i test architektury

```bash
python check_data.py --data data
python test_model.py
```

Test architektury nie wymaga datasetu ani wytrenowanych wag.

## Trening — prototyp 64x64

```bash
python train.py --data data --epochs 10 --batch-size 1 --size 64
```

Wznowienie z ostatniego checkpointu:

```bash
python train.py --data data --epochs 20 --batch-size 1 --size 64 --resume
```

Parametr `--epochs` oznacza docelowa liczbe epok, wiec po wznowieniu trening bedzie kontynuowany do tej liczby.

## Generowanie

```bash
python generate.py --prompt "Fioletowy smok lecacy nad zamkiem" --output generated/test.png --sampling-steps 100 --seed 321
```

Ten sam prompt i seed daja powtarzalny punkt startowy losowania.

## Ograniczenia i status

To nadal eksperymentalny generator badawczy 64x64, a nie narzedzie o jakosci komercyjnych generatorow. W repozytorium nie ma wytrenowanych wag. Sam kod nie tworzy automatycznie zdolnosci generowania — potrzebne sa duze, legalne zbiory obraz-opis i dlugi trening. RTX 2060 6 GB mocno ogranicza rozmiar modelu, batch i rozdzielczosc. Najpierw nalezy uruchomic test architektury, sprawdzic dane, a nastepnie zrobic krotki eksperyment treningowy i ocenic wyniki.

Checkpointy, obrazy i lokalne dane sa ignorowane przez Git. Nie publikuj ich przypadkowo.
