"""
EduHinglish — English NLP Preprocessing Pipeline
==================================================
Author  : Ashvatth
Module  : M1 — Input Processing
Purpose : Complete NLP preprocessing for English NCERT text.
          Every step is isolated, traceable, and saves input→output.

Pipeline:
    Sentence Segmentation → Tokenization → Stop Word Removal →
    Stemming → Lemmatization → POS Tagging (Subject/Object/Verb)

Usage:
    python preprocessing.py                  # runs built-in demo
    python preprocessing.py --text "..."     # custom sentence
"""

import re
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# NLTK imports with graceful fallback
# ─────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, LancasterStemmer
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARN] NLTK not installed. Run: pip install nltk")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("[WARN] spaCy not installed. Run: pip install spacy")


# ─────────────────────────────────────────────────────────────
# POS tag → human-readable description
# ─────────────────────────────────────────────────────────────
POS_DESCRIPTIONS = {
    'NN'  : 'Noun (singular)',
    'NNS' : 'Noun (plural)',
    'NNP' : 'Proper Noun (singular)',
    'NNPS': 'Proper Noun (plural)',
    'VB'  : 'Verb (base form)',
    'VBD' : 'Verb (past tense)',
    'VBG' : 'Verb (gerund / -ing)',
    'VBN' : 'Verb (past participle)',
    'VBP' : 'Verb (non-3rd person present)',
    'VBZ' : 'Verb (3rd person present)',
    'JJ'  : 'Adjective',
    'JJR' : 'Adjective (comparative)',
    'JJS' : 'Adjective (superlative)',
    'RB'  : 'Adverb',
    'RBR' : 'Adverb (comparative)',
    'IN'  : 'Preposition / subordinating conjunction',
    'DT'  : 'Determiner',
    'CC'  : 'Coordinating conjunction',
    'PRP' : 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'MD'  : 'Modal verb',
    'CD'  : 'Cardinal number',
    'WP'  : 'Wh-pronoun',
    'WRB' : 'Wh-adverb',
}


class EnglishPreprocessor:
    """
    Complete NLP preprocessing pipeline for English NCERT Biology text.

    Each public method corresponds to one step in the pipeline and
    prints a clear step header with input/output so the mentor can
    see exactly what is happening.
    """

    # ──────────────────────────────────────────────────────────
    # INITIALISATION
    # ──────────────────────────────────────────────────────────

    def __init__(self):
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required. Run: pip install nltk")

        self.stemmer_porter    = PorterStemmer()
        self.stemmer_lancaster = LancasterStemmer()
        self.lemmatizer        = WordNetLemmatizer()
        self.stop_words        = set(stopwords.words('english'))

        # ── Domain-specific additions to stop list ──
        domain_stops = {
            'also', 'however', 'therefore', 'thus', 'hence',
            'figure', 'fig', 'table', 'example', 'following',
            'given', 'shown', 'see', 'note', 'refer', 'activity',
            'chapter', 'page', 'section',
        }
        self.stop_words.update(domain_stops)

        # ── Words that must NEVER be removed ──
        # (negations and quantifiers carry meaning in science)
        self.protected_words = {
            'not', 'no', 'nor', 'never', 'neither',
            'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'between', 'through', 'into', 'within', 'without',
        }

        # ── spaCy model (optional but preferred) ──
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                print("[OK] spaCy en_core_web_sm loaded")
            except OSError:
                print("[WARN] spaCy model missing. Run: python -m spacy download en_core_web_sm")

        print("[OK] EnglishPreprocessor initialized")

    # ──────────────────────────────────────────────────────────
    # STEP 1 — SENTENCE SEGMENTATION
    # ──────────────────────────────────────────────────────────

    def segment_sentences(self, text: str) -> list:
        """
        Split a block of text into individual sentences.

        Uses spaCy if available (better at handling abbreviations like
        'Fig.', 'e.g.', 'Dr.'); falls back to NLTK sent_tokenize.

        Returns: list of sentence strings
        """
        print(f"\n{'='*60}")
        print("STEP 1 — SENTENCE SEGMENTATION")
        print(f"{'='*60}")
        print(f"  Input  : {len(text):,} characters")

        # NLTK (always available)
        sentences_nltk = sent_tokenize(text)

        # spaCy (preferred)
        if self.nlp:
            doc = self.nlp(text)
            sentences_spacy = [s.text.strip() for s in doc.sents]
        else:
            sentences_spacy = []

        # Pick best available
        sentences_raw = sentences_spacy if sentences_spacy else sentences_nltk

        # Filter out very short fragments (page numbers, artefacts)
        sentences = [s.strip() for s in sentences_raw if len(s.strip()) > 15]

        print(f"  NLTK segments  : {len(sentences_nltk)}")
        if sentences_spacy:
            print(f"  spaCy segments : {len(sentences_spacy)}")
        print(f"  Final (clean)  : {len(sentences)}")

        print(f"\n  First 3 sentences:")
        for i, s in enumerate(sentences[:3], 1):
            preview = s[:100] + ('...' if len(s) > 100 else '')
            print(f"    [{i}] {preview}")

        return sentences

    # ──────────────────────────────────────────────────────────
    # STEP 2 — TOKENIZATION
    # ──────────────────────────────────────────────────────────

    def tokenize(self, sentence: str) -> list:
        """
        Split a sentence into word-level tokens.

        Compares three methods so you can see the difference:
            1. Simple str.split()             — naive
            2. NLTK word_tokenize()           — handles punctuation
            3. spaCy tokenizer                — handles edge cases

        Uses NLTK as the default for this pipeline.

        Returns: list of token strings
        """
        print(f"\n{'='*60}")
        print("STEP 2 — TOKENIZATION")
        print(f"{'='*60}")
        print(f"  Input: \"{sentence}\"")

        # Method 1
        t_simple = sentence.split()

        # Method 2
        t_nltk = word_tokenize(sentence)

        # Method 3
        t_spacy = []
        if self.nlp:
            doc = self.nlp(sentence)
            t_spacy = [tok.text for tok in doc]

        print(f"\n  Simple split   ({len(t_simple):>3} tokens): {t_simple}")
        print(f"  NLTK tokenize  ({len(t_nltk):>3} tokens): {t_nltk}")
        if t_spacy:
            print(f"  spaCy tokenize ({len(t_spacy):>3} tokens): {t_spacy}")

        tokens = t_nltk
        print(f"\n  → Using NLTK tokens: {tokens}")
        return tokens

    # ──────────────────────────────────────────────────────────
    # STEP 3 — STOP WORD REMOVAL
    # ──────────────────────────────────────────────────────────

    def remove_stop_words(self, tokens: list) -> tuple:
        """
        Remove common words that carry little content meaning.

        Protected words (negations, quantifiers) are never removed
        even if they appear in NLTK's stop word list.

        Returns: (kept_tokens, removed_tokens)
        """
        print(f"\n{'='*60}")
        print("STEP 3 — STOP WORD REMOVAL")
        print(f"{'='*60}")
        print(f"  Input tokens ({len(tokens)}): {tokens}")

        kept    = []
        removed = []

        for token in tokens:
            lower = token.lower()

            # Always keep protected words
            if lower in self.protected_words:
                kept.append(token)
                continue

            # Always remove lone punctuation
            if not token.isalnum():
                removed.append(token)
                continue

            # Remove if stop word
            if lower in self.stop_words:
                removed.append(token)
            else:
                kept.append(token)

        print(f"\n  Removed ({len(removed)}): {removed}")
        print(f"  Kept    ({len(kept)})   : {kept}")
        return kept, removed

    # ──────────────────────────────────────────────────────────
    # STEP 4 — STEMMING
    # ──────────────────────────────────────────────────────────

    def stem_tokens(self, tokens: list) -> tuple:
        """
        Reduce words to their root form by aggressively stripping
        suffixes (does NOT guarantee a real dictionary word).

        Porter   — conservative, standard for English IR
        Lancaster — aggressive, produces shorter stems

        Examples (biology context):
            'organelles'  → Porter: 'organel'   Lancaster: 'organ'
            'selectively' → Porter: 'select'    Lancaster: 'select'
            'permeable'   → Porter: 'permeabl'  Lancaster: 'perm'

        Returns: (porter_stems, lancaster_stems)
        """
        print(f"\n{'='*60}")
        print("STEP 4 — STEMMING")
        print(f"{'='*60}")
        print(f"  Input: {tokens}\n")

        porter_stems    = []
        lancaster_stems = []

        col_w = max(len(t) for t in tokens) + 2 if tokens else 20
        col_w = max(col_w, 20)

        header = f"  {'Token':<{col_w}} {'Porter':<{col_w}} {'Lancaster':<{col_w}}"
        print(header)
        print(f"  {'-'*col_w} {'-'*col_w} {'-'*col_w}")

        for token in tokens:
            p = self.stemmer_porter.stem(token)
            l = self.stemmer_lancaster.stem(token)
            porter_stems.append(p)
            lancaster_stems.append(l)

            p_mark = ' ←' if p != token.lower() else ''
            l_mark = ' ←' if l != token.lower() else ''
            print(f"  {token:<{col_w}} {p+p_mark:<{col_w}} {l+l_mark:<{col_w}}")

        print(f"\n  → Porter output: {porter_stems}")
        return porter_stems, lancaster_stems

    # ──────────────────────────────────────────────────────────
    # STEP 5 — LEMMATIZATION
    # ──────────────────────────────────────────────────────────

    def lemmatize_tokens(self, tokens: list) -> list:
        """
        Reduce words to their dictionary base form (lemma).
        Unlike stemming, results are always real words.

        Tries all three POS categories (noun, verb, adjective) and
        picks the shortest (most-reduced) result.

        Examples (biology context):
            'organelles' → 'organelle'
            'carrying'   → 'carry'
            'permeable'  → 'permeable'  (no change — already base)
            'cells'      → 'cell'

        Returns: list of lemma strings
        """
        print(f"\n{'='*60}")
        print("STEP 5 — LEMMATIZATION")
        print(f"{'='*60}")
        print(f"  Input: {tokens}\n")

        lemmas = []
        col_w  = max(len(t) for t in tokens) + 2 if tokens else 20
        col_w  = max(col_w, 20)

        print(f"  {'Token':<{col_w}} {'As Noun':<{col_w}} {'As Verb':<{col_w}} {'Best':<{col_w}}")
        print(f"  {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

        for token in tokens:
            noun_l = self.lemmatizer.lemmatize(token.lower(), pos='n')
            verb_l = self.lemmatizer.lemmatize(token.lower(), pos='v')
            adj_l  = self.lemmatizer.lemmatize(token.lower(), pos='a')

            best = min([noun_l, verb_l, adj_l], key=len)
            lemmas.append(best)

            marker = ' ←' if best != token.lower() else ''
            print(f"  {token:<{col_w}} {noun_l:<{col_w}} {verb_l:<{col_w}} {best+marker}")

        print(f"\n  → Lemmatized output: {lemmas}")
        return lemmas

    # ──────────────────────────────────────────────────────────
    # STEP 6 — POS TAGGING
    # ──────────────────────────────────────────────────────────

    def pos_tag(self, tokens: list) -> tuple:
        """
        Part-of-Speech (POS) tagging — assigns grammatical role to each token.

        This is what the mentor means by 'Subject Object Word' analysis.
        Identifies:
            Nouns  (NN*)  → potential subjects and objects
            Verbs  (VB*)  → actions / processes
            Adjectives (JJ*) → descriptors / properties

        Returns: (pos_tag_list, categories_dict)
        """
        print(f"\n{'='*60}")
        print("STEP 6 — POS TAGGING  (Subject / Object / Verb Analysis)")
        print(f"{'='*60}")
        print(f"  Input: {tokens}\n")

        pos_tags   = nltk.pos_tag(tokens)
        col_w      = max(len(t) for t in tokens) + 2 if tokens else 20
        col_w      = max(col_w, 20)

        print(f"  {'Token':<{col_w}} {'POS':<8} Description")
        print(f"  {'-'*col_w} {'-'*8} {'-'*35}")

        nouns       = []
        verbs       = []
        adjectives  = []

        for word, tag in pos_tags:
            desc = POS_DESCRIPTIONS.get(tag, f'Other ({tag})')
            print(f"  {word:<{col_w}} {tag:<8} {desc}")

            if tag.startswith('NN'):
                nouns.append(word)
            elif tag.startswith('VB'):
                verbs.append(word)
            elif tag.startswith('JJ'):
                adjectives.append(word)

        categories = {
            'nouns'      : nouns,
            'verbs'      : verbs,
            'adjectives' : adjectives,
        }
        print(f"\n  Nouns (subject/object candidates) : {nouns}")
        print(f"  Verbs (actions/processes)          : {verbs}")
        print(f"  Adjectives (properties)            : {adjectives}")

        return pos_tags, categories

    # ──────────────────────────────────────────────────────────
    # COMPLETE PIPELINE — single sentence
    # ──────────────────────────────────────────────────────────

    def process_sentence(self, sentence: str) -> dict:
        """
        Run all pipeline steps on ONE sentence.
        Returns a dictionary with full trace of every step.
        """
        print(f"\n{'#'*70}")
        print(f"# PROCESSING SENTENCE")
        print(f"# \"{sentence[:70]}{'...' if len(sentence)>70 else ''}\"")
        print(f"{'#'*70}")

        result = {"original": sentence, "steps": {}}

        tokens                               = self.tokenize(sentence)
        result["steps"]["tokenization"]      = tokens

        kept, removed                        = self.remove_stop_words(tokens)
        result["steps"]["stop_word_removal"] = {"kept": kept, "removed": removed}

        porter, lancaster                    = self.stem_tokens(kept)
        result["steps"]["stemming"]          = {"porter": porter, "lancaster": lancaster}

        lemmas                               = self.lemmatize_tokens(kept)
        result["steps"]["lemmatization"]     = lemmas

        pos_tags, categories                 = self.pos_tag(kept)
        result["steps"]["pos_tagging"]       = {
            "tags"      : [(w, t) for w, t in pos_tags],
            "categories": categories,
        }

        return result

    # ──────────────────────────────────────────────────────────
    # COMPLETE PIPELINE — full text
    # ──────────────────────────────────────────────────────────

    def process_text(self, text: str, max_sentences: int = None) -> list:
        """
        Run the full pipeline on a block of text.
        Segments into sentences first, then processes each one.
        """
        sentences = self.segment_sentences(text)

        if max_sentences:
            sentences = sentences[:max_sentences]
            print(f"\n[INFO] Capped at {max_sentences} sentences as requested")

        results = []
        for i, sentence in enumerate(sentences, 1):
            print(f"\n\n{'*'*70}")
            print(f"* SENTENCE {i} / {len(sentences)}")
            print(f"{'*'*70}")
            res = self.process_sentence(sentence)
            res["sentence_number"] = i
            results.append(res)

        return results

    # ──────────────────────────────────────────────────────────
    # SAVE
    # ──────────────────────────────────────────────────────────

    def save_results(self, results, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        serializable = json.loads(json.dumps(results, default=str, ensure_ascii=False))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] Results → {output_path}")


# ─────────────────────────────────────────────────────────────
# CLI / direct run
# ─────────────────────────────────────────────────────────────

DEMO_SENTENCES = [
    # Classic biology sentence — tests domain stop-word removal
    "All living organisms are made up of cells, which are the fundamental "
    "structural and functional units of life.",

    # Tests lemmatization: 'permeable', 'controls', 'substances'
    "The cell membrane is a selectively permeable membrane that controls "
    "the movement of substances into and out of the cell.",
]


def main():
    parser = argparse.ArgumentParser(description="EduHinglish English Preprocessor")
    parser.add_argument("--text", type=str, default=None, help="Custom sentence to process")
    parser.add_argument("--file", type=str, default=None, help="Path to cleaned text file")
    parser.add_argument("--max-sentences", type=int, default=2)
    args = parser.parse_args()

    preprocessor = EnglishPreprocessor()

    if args.text:
        result = preprocessor.process_sentence(args.text)

    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        results = preprocessor.process_text(text, max_sentences=args.max_sentences)
        out = Path(__file__).parent.parent / "outputs" / "pipeline_results" / "english_preprocessing.json"
        preprocessor.save_results(results, str(out))

    else:
        print("\n" + "="*70)
        print("DEMO — Processing 2 NCERT Biology sentences")
        print("="*70)
        for sent in DEMO_SENTENCES:
            preprocessor.process_sentence(sent)


if __name__ == "__main__":
    main()