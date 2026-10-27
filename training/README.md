# Dostrajanie Purple AI przez LoRA

To narzędzie dostraja istniejący model bazowy za pomocą adaptera LoRA. Nie trenuje modelu od zera i nie gwarantuje automatycznej poprawy każdej umiejętności.

## Przygotowanie danych
1. Skopiuj `training/data.example.jsonl` do `training/data.jsonl`.
2. Zastąp przykłady własnymi, sprawdzonymi pytaniami i dobrymi odpowiedziami.
3. Każdy wiersz musi być osobnym obiektem JSON z polem `messages`, np.:
   `{"messages":[{"role":"user","content":"Pytanie"},{"role":"assistant","content":"Dobra odpowiedź"}]}`
4. Używaj wyłącznie danych, do których masz prawo. Usuń dane osobowe, sekrety i błędne odpowiedzi. Więcej dobrych przykładów jest zwykle lepsze niż dużo słabych.

## Uruchomienie
W środowisku Python 3.11+ z kompatybilnym GPU i odpowiednią pamięcią:

```bash
pip install -r requirements-model.txt
cp training/data.example.jsonl training/data.jsonl
# Edytuj training/data.jsonl i dodaj własne przykłady
python training/train_lora.py
```

Wynik domyślnie trafia do `purple-ai-lora/`. Aby uruchomić API z adapterem w tym samym środowisku, ustaw `MODEL_ID=purple-ai-lora` i uruchom serwer. Sprawdź kompatybilność adaptera i bazowego modelu. Nie uruchamiaj treningu na zwykłym hostingu webowym bez wymaganej pamięci GPU.

## Ważne
- To szkielet treningu — nie został przetestowany na GPU w ramach tej zmiany.
- Zacznij od małej próbki i sprawdź odpowiedzi przed dalszym treningiem.
- Zapisuj kopię modelu bazowego i porównuj wyniki przed/po; trening może pogorszyć część umiejętności.
