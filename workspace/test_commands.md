# ── 1. Full evaluation (all hardcoded test cases) ──
python training/evaluate_models.py
# ── 2. Quick one-shot test (all 3 models on one sentence) ──
python training/test_models_quick.py --model all --text "Sir, chloroplast ka function kya hai?"
# ── 3. Test a specific model only ──
python training/test_models_quick.py --model lid --text "Mitochondria ko powerhouse kehte hain"
python training/test_models_quick.py --model ner --text "Mendel ne pea plants par experiments kiye"
python training/test_models_quick.py --model intent --text "Photosynthesis aur respiration mein kya fark hai?"
# ── 4. Interactive REPL mode (type sentences, see results live) ──
python training/test_models_quick.py --model all      # all 3 models
python training/test_models_quick.py --model lid       # just LID
python training/test_models_quick.py --model ner       # just NER
python training/test_models_quick.py --model intent    # just Intent
# ── 5. Load a model directly in Python ──
python -c "
import spacy
nlp = spacy.load('models/lid_v1')
doc = nlp('Nucleus cell ka control centre hai')
for tok in doc:
    print(f'{tok.text:15} → {tok.tag_}')
"