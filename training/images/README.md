# Purple AI Image — własny generator obrazów

## Cel

Zbudować własny system generowania obrazów Purple AI, trenowany na własnych wagach i uruchamiany przez serwer Purple AI. Ten projekt nie może wywoływać GPT Image, DALL·E, Stable Diffusion, FLUX ani żadnego zewnętrznego API/modelu obrazowego.

To jest plan techniczny i punkt startowy, a nie działający, wytrenowany generator. Nie wolno oznaczać go jako gotowego, dopóki nie przejdzie treningu i testów jakości.

## Ważne rozdzielenie modeli

- Model tekstowy Purple AI V1.1 pozostaje modelem rozmów.
- Generator obrazów będzie osobnym, autorskim modelem graficznym w tym samym projekcie i API.
- Wspólne API będzie kierować żądanie obrazu do lokalnego modelu Purple AI Image.
- Nie trzeba ponownie trenować modelu tekstowego, aby rozwijać generator obrazów.

## Proponowana architektura od zera

1. **Zbiór danych** — legalnie używalne obrazy z opisami tekstowymi, licencje i źródła zapisane w metadanych. Usuwanie duplikatów, uszkodzonych plików i bardzo małych obrazów.
2. **Tokenizer tekstu** — osobny tokenizer opisów, z obsługą języka polskiego i angielskiego.
3. **Autoenkoder obrazów** — własny encoder/decoder uczący się kompresować obraz do reprezentacji latentnej i odtwarzać go.
4. **Model dyfuzyjny** — własna sieć ucząca się odszumiać reprezentację latentną.
5. **Warunkowanie tekstem** — enkoder tekstu i mechanizm cross-attention, aby opis sterował generowanym obrazem.
6. **Sampler i dekoder** — zamiana wygenerowanej reprezentacji latentnej na PNG.
7. **API** — endpoint obrazu dostępny tylko wtedy, gdy istnieją wytrenowane wagi i model przejdzie test ładowania.

## Kolejność prac

- Etap 1: przygotowanie manifestu danych i walidatora zbioru.
- Etap 2: implementacja oraz testy autoenkodera.
- Etap 3: trening autoenkodera i sprawdzenie rekonstrukcji.
- Etap 4: implementacja i trening modelu dyfuzyjnego.
- Etap 5: połączenie z warunkowaniem tekstowym.
- Etap 6: testy promptów, jakości, bezpieczeństwa i szybkości.
- Etap 7: integracja z API Purple AI.

## Sprzęt i realistyczne oczekiwania

RTX 2060 z 6 GB VRAM może służyć do testów małych komponentów, ale nie jest wystarczająca do szybkiego trenowania dużego generatora obrazów wysokiej jakości od zera. Potrzebny będzie mocniejszy GPU lub wynajęta maszyna obliczeniowa. Mały prototyp może generować proste obrazy, ale jakość porównywalna z dużymi komercyjnymi generatorami wymaga znacznie większych danych, obliczeń i iteracji.

## Zasady projektu

- Nie pobieraj ani nie uruchamiaj wag innego generatora obrazów jako ukrytego zamiennika.
- Nie wysyłaj promptów ani obrazów do zewnętrznego dostawcy.
- Nie dodawaj kluczy dostawców obrazowych do `.env`.
- Nie udawaj generowania: jeśli wagi nie istnieją, API ma zwrócić jasny komunikat `image_model_not_trained`.
- Zapisuj konfigurację, seed, wersję danych i checkpoint, aby wyniki dało się odtworzyć.
