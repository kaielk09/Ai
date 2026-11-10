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

Nowe treningi używają harmonogramu dyfuzji **cosine-v2**. Poprzedni harmonogram liniowy przy 100 krokach zostawiał zbyt dużo oryginalnego obrazu na końcu procesu szumienia (alpha_bar końcowe około 0.36), podczas gdy generator rozpoczynał tworzenie od czystego losowego szumu. To powodowało poważną niezgodność treningu i generowania oraz obrazy przypominające losowe piksele.

Nowy harmonogram cosine-v2 kończy blisko czystego szumu także przy 100 krokach. Nowy checkpoint zapisuje nazwę harmonogramu. Stare checkpointy nadal można odczytać do generowania ze starym harmonogramem, ale **nie wolno wznawiać ich treningu jako cosine-v2** — trzeba rozpocząć nowy trening.

Przykład techniczny:

```bash
python train.py --data data --epochs 10 --batch-size 1 --size 64 --steps 100
```

To tylko test działania kodu. 100 obrazów i jedna epoka (około 100 aktualizacji) nie wystarczą do nauczenia text-to-image. Nawet 1000 par i 10000 aktualizacji to wciąż etap eksperymentalny, nie gwarancja dobrej jakości. Do sensownego treningu od zera potrzebny jest znacznie większy, różnorodny zbiór poprawnie opisanych obrazów i długi trening na GPU. Samo zwiększenie liczby kroków dyfuzji ani rozdzielczości nie zastąpi danych i treningu.

Wznowienie z ostatniego checkpointu:

```bash
python train.py --data data --epochs 20 --batch-size 1 --size 64 --resume
```

Parametr `--epochs` oznacza docelowa liczbe epok, wiec po wznowieniu trening bedzie kontynuowany do tej liczby.

## Generowanie w roznych rozmiarach

Generator pozwala ustawiac osobno szerokosc i wysokosc. Dostepne sa rozne proporcje, nie tylko kwadrat. Przykladowe polecenia:

```bash
# Kwadrat
python generate.py --prompt "Fioletowy smok nad zamkiem" --size 64
python generate.py --prompt "Fioletowy smok nad zamkiem" --size 128
python generate.py --prompt "Fioletowy smok nad zamkiem" --size 256

# Poziomy i pionowy
python generate.py --prompt "Gorskie jezioro" --size 512x384
python generate.py --prompt "Wieza w lesie" --size 384x512

# Jawne wymiary (maja pierwszenstwo przed --size)
python generate.py --prompt "Miasto nocą" --width 512 --height 320 --sampling-steps 100
```

Zakres wynosi 16..1024 pikseli na bok, a laczna powierzchnia jest ograniczona do 1 048 576 pikseli. Można podać dowolne wymiary w tym zakresie, np. 320x240, 768x512 lub 1024x512. Wiekszy rozmiar oznacza wiecej pamieci i wolniejsze generowanie. Model uczony w 64x64 moze technicznie wygenerowac wieksze plotno, ale nie oznacza to automatycznie wiekszej ilosci nauczonych szczegolow. Dla najlepszej jakosci trenuj na rozmiarze docelowym i stopniowo go zwiekszaj.

Ten sam prompt i seed daja powtarzalny punkt startowy losowania.

## Kolory i odcienie\n\nParser promptu rozpoznaje polskie aliasy kolorów, nazwy kolorów CSS (np. `cornflowerblue`, `rebeccapurple`) oraz dowolne wartości `#RGB`, `#RRGGBB`, `#RRGGBBAA` i `rgb(r,g,b)`. Z wykrytych barw buduje paletę odcieni i przekazuje ją jako dodatkowe warunkowanie tekstowe modelu. To nie jest ręczne malowanie pikseli — model musi nauczyć się odwzorowywać kolory podczas treningu.\n\n## Ograniczenia i status

To nadal eksperymentalny generator od zera, a nie generator o jakosci komercyjnej. 300 obrazow to za malo, by nauczyc go wiarygodnego text-to-image; zwykle potrzebny jest duzy, roznorodny i poprawnie opisany zbior oraz dlugi trening na GPU. Samo zwiekszenie rozdzielczosci do 1024 lub 10000 pikseli nie dodaje szczegolow, ktorych model nie nauczyl sie na treningu.

### Harmonogram dyfuzji cosine-v2

Nowe treningi korzystają z harmonogramu cosine-v2. Generator wybiera harmonogram zapisany w checkpointcie: nowe checkpointy używają cosine-v2, a starsze checkpointy bez metadanych zachowują zgodność ze starym harmonogramem liniowym. Po zmianie harmonogramu wymagany jest nowy trening od losowych wag.

### Poprawka EMA

Generator podczas inferencji preferuje wagi EMA. Wczesniej stale EMA=0.999 od pierwszego kroku moglo powodowac, ze po krotkim treningu EMA nadal przypominala losowa inicjalizacje. Trening uzywa teraz rozgrzewki EMA zależnej od liczby aktualizacji i zapisuje licznik EMA w checkpointach. Dodano tez losowe odbicie poziome jako lekka augmentacje. Stare checkpointy mozna wczytac do wznowienia, ale zalecany jest swiezy trening, aby zbudowac nowe wagi EMA.

RTX 2060 6 GB mocno ogranicza rozmiar modelu, batch i rozdzielczosc. Zacznij od 64x64, sprawdz probki, a dopiero potem eksperymentuj z wiekszym rozmiarem. Nie ustawiaj 10000x10000: taki rozmiar jest poza praktycznym zakresem tego prototypu. Uzywaj wylacznie obrazow, do ktorych masz prawa.

Checkpointy, obrazy i lokalne dane sa ignorowane przez Git. Nie publikuj ich przypadkowo.
