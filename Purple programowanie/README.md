# Purple AI V1.0 — moduł programowania

Ten folder dodaje dane i skrypty do uczenia **tego samego modelu Purple AI V1.0** umiejętności związanych z programowaniem. Nie tworzy osobnego modelu ani osobnego API.

## Co robi moduł

1. Zbiera lokalne przykłady kodu z wielu języków programowania.
2. Oznacza każdy przykład nazwą języka i ścieżką pliku.
3. Buduje tekstowy korpus treningowy z fragmentami kodu i prostymi zadaniami opisowymi.
4. Uruchamia istniejący `training/train_from_scratch.py`, korzystając z tej samej architektury i katalogu checkpointów Purple AI.

## Obsługiwane rozszerzenia

Python, JavaScript, TypeScript, JSX, TSX, Java, Kotlin, Swift, C, C++, C#, Go, Rust, Ruby, PHP, Lua, Dart, Scala, R, Julia, Perl, Haskell, Elixir, Erlang, Clojure, F#, Visual Basic, PowerShell, Bash, SQL, HTML, CSS, SCSS, Vue, Svelte, XML, JSON, YAML, TOML, GraphQL, Solidity, Move, Verilog, VHDL, MATLAB, Octave, Fortran, COBOL, Assembly, Markdown i pliki konfiguracyjne.

To lista formatów, które skrypt potrafi rozpoznać — nie oznacza, że model już zna te języki.

## Przygotowanie danych

Umieść własne, legalnie używalne projekty w katalogu, np. `/content/kod-do-nauki`, a następnie:

```bash
python "Purple programowanie/prepare_dataset.py" \
  --source /content/kod-do-nauki \
  --output training/programming_data.txt
```

Można podać kilka katalogów przez powtórzenie `--source`. Skrypt pomija typowe katalogi wynikowe, pliki binarne, sekrety i zbyt duże pliki.

## Trening tego samego Purple AI V1.0

```bash
python "Purple programowanie/train.py" \
  --data training/programming_data.txt \
  --steps 20000 \
  --batch-size 16
```

Domyślnie checkpoint jest zapisywany w istniejącym katalogu `checkpoints/`. Jeśli istnieje zgodny checkpoint Purple AI V1.0, trening zostanie wznowiony. To nadal jeden model i jeden checkpoint.

## Ważne ograniczenia

- Obecny model to mały, znakowo-bajtowy Transformer z kontekstem 512 bajtów. Nie jest jeszcze zaawansowanym asystentem programistycznym.
- Samo wczytanie plików źródłowych uczy głównie przewidywania kolejnych bajtów. Nie gwarantuje rozumienia poleceń, naprawiania błędów ani poprawnego tworzenia całych aplikacji.
- Dobre programowanie wymaga dużego, zróżnicowanego zbioru wysokiej jakości: instrukcja, rozwiązanie, testy, poprawki i wyjaśnienia.
- Nie istnieje realistyczny sposób, aby zagwarantować znajomość wszystkich języków programowania.
- Używaj wyłącznie danych, do których masz prawa. Nie dodawaj kluczy API, haseł, danych prywatnych ani plików z sekretami.
- Trening może nadpisać/zmienić wspólny checkpoint modelu. Zachowaj kopię checkpointu przed rozpoczęciem.
