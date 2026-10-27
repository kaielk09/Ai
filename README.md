# Purple AI v1.0

Polskojęzyczny interfejs czatu AI z konfigurowalnym dostawcą API i opcjonalnym wyszukiwaniem internetowym.

## Wymagania

- Node.js 20 lub nowszy
- Dostawca API obsługujący format OpenAI Chat Completions
- Opcjonalnie klucz Tavily do wyszukiwania stron

## Uruchomienie lokalne

1. Sklonuj repozytorium.
2. Uruchom `npm install`.
3. Ustaw zmienne środowiskowe opisane w `.env.example` (możesz utworzyć lokalny plik `.env`; nigdy go nie publikuj).
4. Uruchom `npm start`.
5. Otwórz `http://localhost:3000`.

## Zmienne środowiskowe

- `API_BASE_URL` — adres API, np. bazowy adres zakończony na `/v1`, bez `/chat/completions`.
- `API_KEY` — prawdziwy klucz dostawcy; ustaw go w sekretach hostingu, nie w kodzie.
- `MODEL_NAME` — dokładna nazwa modelu obsługiwana przez dostawcę.
- `SEARCH_API_KEY` — opcjonalny klucz Tavily do wyszukiwania w sieci.

API musi obsługiwać endpoint `POST /chat/completions` i nagłówek `Authorization: Bearer ...`. Jeśli dostawca ma inny format, trzeba dostosować backend.

## Hosting

To aplikacja Node.js, nie statyczna strona GitHub Pages. Uruchom ją na hostingu obsługującym Node.js i ustaw zmienne środowiskowe w panelu hostingu. Repozytorium GitHub może przechowywać kod; samo GitHub Pages nie uruchomi serwera Node.js.

## Bezpieczeństwo

Nie commituj prawdziwych kluczy API. Ogranicz limity i uprawnienia klucza u dostawcy. Publiczna aplikacja wymaga dodatkowo limitów zapytań, uwierzytelniania i ochrony przed nadużyciami przed udostępnieniem jej wielu użytkownikom.

Purple AI v1.0 korzysta z modelu wskazanego przez dostawcę API; sam kod aplikacji nie tworzy ani nie trenuje nowego modelu od zera.