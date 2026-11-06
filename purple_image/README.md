# Purple AI Image

Wlasny, autorski projekt generatora obrazow Purple AI.

## Zasady projektu

- Nie korzystamy z zewnetrznych API do generowania obrazow.
- Nie pobieramy gotowych generatorow ani gotowych wag modeli obrazowych.
- Architektura generatora, kod treningu, kod inferencji i format wag beda tworzone w tym repozytorium.
- Model tekstowy Qwen pozostaje osobnym modelem tekstowym; nie udajemy, ze sam potrafi generowac obrazy.
- Nie zapisujemy tutaj cudzych wag ani danych treningowych bez sprawdzonej licencji.

## Plan rozwoju

1. Zdefiniowac wlasny format danych: obraz + opis tekstowy.
2. Zaimplementowac autorski kod reprezentacji obrazu i dekodera.
3. Zaimplementowac mechanizm warunkowania generacji opisem.
4. Zaimplementowac trening, zapisywanie wag, wznawianie treningu i inferencje.
5. Dodac testy oraz pomiary jakosci i zuzycia pamieci GPU.
6. Dopiero po testach podlaczyc generator do API Purple AI.

## Status

Na tym etapie folder jest szkieletem projektu. Nie ma jeszcze wytrenowanych wag ani dzialajacego generatora. Pierwsza wersja bedzie eksperymentalna; jakosc bedzie rosla wraz z danymi, treningiem i dostepnym sprzetem.

## Decyzja techniczna przed implementacja

Trzeba ustalic, czy dozwolony jest PyTorch wylacznie jako silnik obliczen tensorowych i obslugi GPU. Nie bedzie uzywany jako gotowy generator ani jako zrodlo gotowych wag. Jesli PyTorch tez jest zakazany, implementacja od zera bedzie znacznie wolniejsza i praktycznie nie wykorzysta RTX 2060 do treningu.
