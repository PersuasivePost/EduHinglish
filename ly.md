



\# EduHinglish — Getting Started: Preprocessing Pipeline

\## Practical Work Plan for Ashvatth \& Jatin



\---



\## UNDERSTANDING WHAT YOUR MENTOR WANTS



Based on your mentor's keywords, here's what they expect you to deliver:



```

MENTOR'S EXPECTATION (Decoded):



1\. Take ONE chapter from NCERT Class 9 Biology (English PDF)

2\. Extract text from the PDF → Clean it

3\. Build a COMPLETE NLP Preprocessing Pipeline:

&#x20;  ├── Script Detection

&#x20;  ├── Normalization  

&#x20;  ├── Tokenization

&#x20;  ├── Stemming

&#x20;  ├── Lemmatization

&#x20;  ├── Stop Word Removal

&#x20;  ├── Sentence Segmentation

&#x20;  └── Language Detection (word-level)



4\. Create Hinglish (code-mixed) versions of sentences from that chapter

5\. Run the pipeline on BOTH English AND Hinglish sentences

6\. Show input vs output at EVERY step

7\. Compare original output with reference output

8\. Present it all

```



\*\*Key mentor instruction: "1-2 sentences pe working" → Start small. Get the pipeline working on 1-2 sentences first, THEN scale to the full chapter.\*\*



\---



\## TASK DIVISION: WHO DOES WHAT



```

ASHVATTH's Tasks:

├── Task A1: NCERT PDF text extraction \& cleaning

├── Task A2: Sentence segmentation

├── Task A3: Tokenization + Stop word removal

├── Task A4: Stemming + Lemmatization

└── Task A5: Building the unified pipeline class



JATIN's Tasks:

├── Task J1: Create Hinglish dataset (code-mixed sentences from the chapter)

├── Task J2: Script detection module

├── Task J3: Normalization module (spelling + script)

├── Task J4: Word-level Language Identification (LID)

└── Task J5: Pipeline integration + testing on Hinglish input



TOGETHER:

├── Compare outputs (English pipeline vs Hinglish pipeline)

├── Document results

└── Prepare presentation

```



\---



\## STEP 0: ENVIRONMENT SETUP (Both do this)



\### Create Project Structure



```

EduHinglish/

│

├── data/

│   ├── raw/

│   │   └── ncert\_class9\_bio\_ch5.pdf      ← NCERT PDF

│   ├── processed/

│   │   ├── chapter\_text\_raw.txt           ← Extracted raw text

│   │   ├── chapter\_text\_clean.txt         ← Cleaned text

│   │   └── chapter\_sentences.json         ← Segmented sentences

│   └── hinglish/

│       └── hinglish\_sentences.json        ← Code-mixed dataset

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── pdf\_extractor.py                   ← Ashvatth

│   ├── preprocessing.py                   ← Ashvatth

│   ├── script\_detector.py                 ← Jatin

│   ├── normalizer.py                      ← Jatin

│   ├── language\_identifier.py             ← Jatin

│   └── pipeline.py                        ← Both

│

├── notebooks/

│   ├── 01\_pdf\_extraction.ipynb            ← Ashvatth

│   ├── 02\_preprocessing\_english.ipynb     ← Ashvatth

│   ├── 03\_hinglish\_dataset.ipynb          ← Jatin

│   ├── 04\_script\_and\_lid.ipynb            ← Jatin

│   └── 05\_full\_pipeline\_demo.ipynb        ← Both

│

├── outputs/

│   └── pipeline\_results/                  ← All outputs saved here

│

├── requirements.txt

└── README.md

```



\### Install Dependencies



```bash

\# Create virtual environment

python -m venv eduhinglish\_env

source eduhinglish\_env/bin/activate  # Linux/Mac

\# OR

eduhinglish\_env\\Scripts\\activate  # Windows



\# Install all required packages

pip install pdfplumber PyPDF2 nltk spacy pandas numpy

pip install indic-transliteration polyglot langdetect

pip install transformers torch sentencepiece

pip install jupyter notebook



\# Download NLTK data

python -c "

import nltk

nltk.download('punkt')

nltk.download('punkt\_tab')

nltk.download('averaged\_perceptron\_tagger')

nltk.download('averaged\_perceptron\_tagger\_eng')

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('omw-1.4')

"



\# Download spaCy English model

python -m spacy download en\_core\_web\_sm

```



\### `requirements.txt`



```

pdfplumber==0.11.0

PyPDF2==3.0.1

nltk==3.9.1

spacy==3.7.4

pandas==2.2.0

numpy==1.26.4

indic-transliteration==2.4.1

langdetect==1.0.9

transformers==4.40.0

torch==2.2.0

sentencepiece==0.2.0

jupyter==1.0.0

```



\---



\## STEP 1: PDF TEXT EXTRACTION (Ashvatth — Task A1)



\### Which Chapter?



For this demo, I'll use \*\*NCERT Class 9 Science — Chapter 5: "The Fundamental Unit of Life"\*\* (Biology section). Download from: https://ncert.nic.in/textbook.php



\### File: `src/pdf\_extractor.py`



```python

"""

EduHinglish - PDF Text Extractor

Author: Ashvatth

Purpose: Extract clean text from NCERT Biology PDF chapter

"""



import pdfplumber

import re

import json

import os





class NCERTPDFExtractor:

&#x20;   """Extract and clean text from NCERT textbook PDFs."""

&#x20;   

&#x20;   def \_\_init\_\_(self, pdf\_path):

&#x20;       self.pdf\_path = pdf\_path

&#x20;       self.raw\_text = ""

&#x20;       self.clean\_text = ""

&#x20;       self.pages\_data = \[]

&#x20;   

&#x20;   def extract\_text(self):

&#x20;       """Extract text from all pages of the PDF."""

&#x20;       print(f"\[INFO] Extracting text from: {self.pdf\_path}")

&#x20;       

&#x20;       with pdfplumber.open(self.pdf\_path) as pdf:

&#x20;           for page\_num, page in enumerate(pdf.pages, 1):

&#x20;               page\_text = page.extract\_text()

&#x20;               if page\_text:

&#x20;                   self.pages\_data.append({

&#x20;                       "page\_number": page\_num,

&#x20;                       "raw\_text": page\_text

&#x20;                   })

&#x20;                   self.raw\_text += page\_text + "\\n\\n"

&#x20;                   print(f"  \[OK] Page {page\_num}: {len(page\_text)} characters extracted")

&#x20;               else:

&#x20;                   print(f"  \[WARN] Page {page\_num}: No text found (might be an image)")

&#x20;       

&#x20;       print(f"\\n\[INFO] Total pages processed: {len(self.pages\_data)}")

&#x20;       print(f"\[INFO] Total raw characters: {len(self.raw\_text)}")

&#x20;       return self.raw\_text

&#x20;   

&#x20;   def clean\_extracted\_text(self, text=None):

&#x20;       """Clean the extracted text by removing noise."""

&#x20;       if text is None:

&#x20;           text = self.raw\_text

&#x20;       

&#x20;       print("\\n\[INFO] Cleaning extracted text...")

&#x20;       

&#x20;       # Step 1: Remove page headers/footers

&#x20;       # NCERT books typically have "THE FUNDAMENTAL UNIT OF LIFE" or chapter number

&#x20;       text = re.sub(

&#x20;           r'(?i)(the fundamental unit of life|chapter\\s\*\\d+|science|class\\s\*(ix|9))',

&#x20;           '', text

&#x20;       )

&#x20;       print("  \[OK] Removed headers/footers")

&#x20;       

&#x20;       # Step 2: Remove page numbers (standalone numbers on a line)

&#x20;       text = re.sub(r'^\\s\*\\d{1,3}\\s\*$', '', text, flags=re.MULTILINE)

&#x20;       print("  \[OK] Removed page numbers")

&#x20;       

&#x20;       # Step 3: Remove excessive whitespace

&#x20;       text = re.sub(r'\\n{3,}', '\\n\\n', text)  # Multiple newlines → double newline

&#x20;       text = re.sub(r'\[ \\t]{2,}', ' ', text)   # Multiple spaces → single space

&#x20;       print("  \[OK] Cleaned whitespace")

&#x20;       

&#x20;       # Step 4: Remove figure/table references but keep surrounding text

&#x20;       text = re.sub(r'Fig\\.\\s\*\\d+\\.\\d+\\s\*:?\\s\*', '', text)

&#x20;       text = re.sub(r'Table\\s\*\\d+\\.\\d+\\s\*:?\\s\*', '', text)

&#x20;       print("  \[OK] Removed figure/table references")

&#x20;       

&#x20;       # Step 5: Fix common OCR/extraction artifacts

&#x20;       text = re.sub(r'(\[a-z])\\n(\[a-z])', r'\\1 \\2', text)  # Fix broken words

&#x20;       text = re.sub(r'(\\w)-\\n(\\w)', r'\\1\\2', text)          # Fix hyphenation

&#x20;       print("  \[OK] Fixed extraction artifacts")

&#x20;       

&#x20;       # Step 6: Normalize quotes and special characters

&#x20;       text = text.replace('"', '"').replace('"', '"')

&#x20;       text = text.replace(''', "'").replace(''', "'")

&#x20;       text = text.replace('–', '-').replace('—', '-')

&#x20;       print("  \[OK] Normalized special characters")

&#x20;       

&#x20;       # Step 7: Remove any remaining artifacts (adjust based on your PDF)

&#x20;       text = re.sub(r'\[^\\w\\s\\.\\,\\;\\:\\!\\?\\-\\(\\)\\\[\\]\\"\\'\\%\\°\\/\\+\\=]', '', text)

&#x20;       print("  \[OK] Removed remaining artifacts")

&#x20;       

&#x20;       self.clean\_text = text.strip()

&#x20;       print(f"\\n\[INFO] Clean text length: {len(self.clean\_text)} characters")

&#x20;       

&#x20;       return self.clean\_text

&#x20;   

&#x20;   def save\_raw\_text(self, output\_path):

&#x20;       """Save raw extracted text to file."""

&#x20;       os.makedirs(os.path.dirname(output\_path), exist\_ok=True)

&#x20;       with open(output\_path, 'w', encoding='utf-8') as f:

&#x20;           f.write(self.raw\_text)

&#x20;       print(f"\[SAVED] Raw text → {output\_path}")

&#x20;   

&#x20;   def save\_clean\_text(self, output\_path):

&#x20;       """Save cleaned text to file."""

&#x20;       os.makedirs(os.path.dirname(output\_path), exist\_ok=True)

&#x20;       with open(output\_path, 'w', encoding='utf-8') as f:

&#x20;           f.write(self.clean\_text)

&#x20;       print(f"\[SAVED] Clean text → {output\_path}")

&#x20;   

&#x20;   def save\_pages\_data(self, output\_path):

&#x20;       """Save page-by-page data as JSON."""

&#x20;       os.makedirs(os.path.dirname(output\_path), exist\_ok=True)

&#x20;       with open(output\_path, 'w', encoding='utf-8') as f:

&#x20;           json.dump(self.pages\_data, f, indent=2, ensure\_ascii=False)

&#x20;       print(f"\[SAVED] Pages data → {output\_path}")

&#x20;   

&#x20;   def get\_text\_stats(self):

&#x20;       """Print statistics about the extracted text."""

&#x20;       text = self.clean\_text if self.clean\_text else self.raw\_text

&#x20;       words = text.split()

&#x20;       sentences = re.split(r'\[.!?]+', text)

&#x20;       paragraphs = \[p for p in text.split('\\n\\n') if p.strip()]

&#x20;       

&#x20;       print("\\n" + "="\*50)

&#x20;       print("TEXT STATISTICS")

&#x20;       print("="\*50)

&#x20;       print(f"  Total characters  : {len(text)}")

&#x20;       print(f"  Total words       : {len(words)}")

&#x20;       print(f"  Total sentences   : {len(sentences)}")

&#x20;       print(f"  Total paragraphs  : {len(paragraphs)}")

&#x20;       print(f"  Avg words/sentence: {len(words)/max(len(sentences),1):.1f}")

&#x20;       print("="\*50)

&#x20;       

&#x20;       return {

&#x20;           "characters": len(text),

&#x20;           "words": len(words),

&#x20;           "sentences": len(sentences),

&#x20;           "paragraphs": len(paragraphs)

&#x20;       }





\# ============================================

\# USAGE — Run this to extract your chapter

\# ============================================

if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   # Change this path to YOUR PDF file

&#x20;   PDF\_PATH = "../data/raw/ncert\_class9\_bio\_ch5.pdf"

&#x20;   

&#x20;   # Initialize extractor

&#x20;   extractor = NCERTPDFExtractor(PDF\_PATH)

&#x20;   

&#x20;   # Extract

&#x20;   raw = extractor.extract\_text()

&#x20;   

&#x20;   # Clean

&#x20;   clean = extractor.clean\_extracted\_text()

&#x20;   

&#x20;   # Save

&#x20;   extractor.save\_raw\_text("../data/processed/chapter\_text\_raw.txt")

&#x20;   extractor.save\_clean\_text("../data/processed/chapter\_text\_clean.txt")

&#x20;   extractor.save\_pages\_data("../data/processed/chapter\_pages.json")

&#x20;   

&#x20;   # Stats

&#x20;   extractor.get\_text\_stats()

&#x20;   

&#x20;   # Preview first 500 characters of clean text

&#x20;   print("\\n\[PREVIEW] First 500 chars of clean text:")

&#x20;   print("-"\*50)

&#x20;   print(clean\[:500])

```



\---



\## STEP 2: NLP PREPROCESSING PIPELINE (Ashvatth — Tasks A2-A5)



\### File: `src/preprocessing.py`



```python

"""

EduHinglish - NLP Preprocessing Pipeline

Author: Ashvatth

Purpose: Complete NLP preprocessing for English text from NCERT

Pipeline: Sentence Segmentation → Tokenization → Stop Word Removal → 

&#x20;         Stemming → Lemmatization → POS Tagging

"""



import re

import json

import nltk

import spacy

from nltk.tokenize import word\_tokenize, sent\_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer, LancasterStemmer

from nltk.stem import WordNetLemmatizer





class EnglishPreprocessor:

&#x20;   """

&#x20;   Complete NLP preprocessing pipeline for English NCERT text.

&#x20;   Each step is separate and traceable — you can see input/output at every stage.

&#x20;   """

&#x20;   

&#x20;   def \_\_init\_\_(self):

&#x20;       # Initialize NLP tools

&#x20;       self.stemmer\_porter = PorterStemmer()

&#x20;       self.stemmer\_lancaster = LancasterStemmer()

&#x20;       self.lemmatizer = WordNetLemmatizer()

&#x20;       self.stop\_words = set(stopwords.words('english'))

&#x20;       

&#x20;       # Load spaCy model for advanced processing

&#x20;       try:

&#x20;           self.nlp = spacy.load('en\_core\_web\_sm')

&#x20;           print("\[OK] spaCy model loaded")

&#x20;       except OSError:

&#x20;           print("\[WARN] spaCy model not found. Run: python -m spacy download en\_core\_web\_sm")

&#x20;           self.nlp = None

&#x20;       

&#x20;       # Custom stop words to ADD (educational domain)

&#x20;       self.custom\_stop\_words = {

&#x20;           'also', 'however', 'therefore', 'thus', 'hence',

&#x20;           'figure', 'fig', 'table', 'example', 'following',

&#x20;           'given', 'shown', 'see', 'note', 'refer'

&#x20;       }

&#x20;       self.stop\_words.update(self.custom\_stop\_words)

&#x20;       

&#x20;       # Words to PROTECT from stop word removal (important science terms 

&#x20;       # that might be removed)

&#x20;       self.protected\_words = {

&#x20;           'not', 'no', 'nor',  # Negations are important in science!

&#x20;           'all', 'each', 'every',  # Quantifiers matter

&#x20;           'between', 'through', 'into', 'within',  # Spatial terms for biology

&#x20;       }

&#x20;       

&#x20;       print("\[OK] English Preprocessor initialized")

&#x20;   

&#x20;   # =========================================

&#x20;   # STEP 1: SENTENCE SEGMENTATION

&#x20;   # =========================================

&#x20;   def segment\_sentences(self, text):

&#x20;       """

&#x20;       Split text into individual sentences.

&#x20;       

&#x20;       Why this matters: Each sentence is a unit of meaning.

&#x20;       The retrieval system needs sentence-level granularity.

&#x20;       """

&#x20;       print("\\n" + "="\*60)

&#x20;       print("STEP 1: SENTENCE SEGMENTATION")

&#x20;       print("="\*60)

&#x20;       

&#x20;       # Method 1: NLTK sentence tokenizer (rule-based)

&#x20;       sentences\_nltk = sent\_tokenize(text)

&#x20;       

&#x20;       # Method 2: spaCy sentence segmentation (model-based, more accurate)

&#x20;       sentences\_spacy = \[]

&#x20;       if self.nlp:

&#x20;           doc = self.nlp(text)

&#x20;           sentences\_spacy = \[sent.text.strip() for sent in doc.sents]

&#x20;       

&#x20;       # Use spaCy if available (better at handling abbreviations like 

&#x20;       # "Dr.", "Fig.", "e.g.")

&#x20;       sentences = sentences\_spacy if sentences\_spacy else sentences\_nltk

&#x20;       

&#x20;       # Clean each sentence

&#x20;       clean\_sentences = \[]

&#x20;       for sent in sentences:

&#x20;           sent = sent.strip()

&#x20;           if len(sent) > 10:  # Skip very short fragments

&#x20;               clean\_sentences.append(sent)

&#x20;       

&#x20;       print(f"  Input text length: {len(text)} chars")

&#x20;       print(f"  Sentences found (NLTK): {len(sentences\_nltk)}")

&#x20;       if sentences\_spacy:

&#x20;           print(f"  Sentences found (spaCy): {len(sentences\_spacy)}")

&#x20;       print(f"  Clean sentences (final): {len(clean\_sentences)}")

&#x20;       

&#x20;       # Show first 3 sentences as preview

&#x20;       print(f"\\n  Preview (first 3 sentences):")

&#x20;       for i, sent in enumerate(clean\_sentences\[:3], 1):

&#x20;           print(f"    \[{i}] {sent\[:100]}{'...' if len(sent) > 100 else ''}")

&#x20;       

&#x20;       return clean\_sentences

&#x20;   

&#x20;   # =========================================

&#x20;   # STEP 2: TOKENIZATION

&#x20;   # =========================================

&#x20;   def tokenize(self, sentence):

&#x20;       """

&#x20;       Split a sentence into individual words (tokens).

&#x20;       

&#x20;       Why this matters: NLP works at the word level.

&#x20;       Each token will be analyzed for language, stemmed, etc.

&#x20;       """

&#x20;       print("\\n" + "="\*60)

&#x20;       print("STEP 2: TOKENIZATION")

&#x20;       print("="\*60)

&#x20;       

&#x20;       print(f"  Input: \\"{sentence}\\"")

&#x20;       

&#x20;       # Method 1: Simple split (naive)

&#x20;       tokens\_simple = sentence.split()

&#x20;       

&#x20;       # Method 2: NLTK word tokenizer (handles punctuation properly)

&#x20;       tokens\_nltk = word\_tokenize(sentence)

&#x20;       

&#x20;       # Method 3: spaCy tokenizer (best — handles edge cases)

&#x20;       tokens\_spacy = \[]

&#x20;       if self.nlp:

&#x20;           doc = self.nlp(sentence)

&#x20;           tokens\_spacy = \[token.text for token in doc]

&#x20;       

&#x20;       # Use NLTK as primary (good balance of speed and accuracy)

&#x20;       tokens = tokens\_nltk

&#x20;       

&#x20;       print(f"\\n  Simple split  ({len(tokens\_simple)} tokens): {tokens\_simple}")

&#x20;       print(f"  NLTK tokenize ({len(tokens\_nltk)} tokens): {tokens\_nltk}")

&#x20;       if tokens\_spacy:

&#x20;           print(f"  spaCy tokenize ({len(tokens\_spacy)} tokens): {tokens\_spacy}")

&#x20;       

&#x20;       print(f"\\n  → Using NLTK tokens: {tokens}")

&#x20;       

&#x20;       return tokens

&#x20;   

&#x20;   # =========================================

&#x20;   # STEP 3: STOP WORD REMOVAL

&#x20;   # =========================================

&#x20;   def remove\_stop\_words(self, tokens):

&#x20;       """

&#x20;       Remove common words that don't carry meaning (the, is, a, etc.)

&#x20;       BUT protect important scientific/negation words.

&#x20;       

&#x20;       Why this matters: Reduces noise for retrieval.

&#x20;       "The cell is the fundamental unit" → "cell fundamental unit"

&#x20;       """

&#x20;       print("\\n" + "="\*60)

&#x20;       print("STEP 3: STOP WORD REMOVAL")

&#x20;       print("="\*60)

&#x20;       

&#x20;       print(f"  Input tokens ({len(tokens)}): {tokens}")

&#x20;       

&#x20;       filtered\_tokens = \[]

&#x20;       removed\_tokens = \[]

&#x20;       

&#x20;       for token in tokens:

&#x20;           token\_lower = token.lower()

&#x20;           

&#x20;           # Keep if it's a protected word

&#x20;           if token\_lower in self.protected\_words:

&#x20;               filtered\_tokens.append(token)

&#x20;               continue

&#x20;           

&#x20;           # Keep if it's NOT a stop word

&#x20;           if token\_lower not in self.stop\_words:

&#x20;               # Also keep if it's a punctuation or number

&#x20;               if token.isalpha() or token.replace('.', '').isdigit():

&#x20;                   filtered\_tokens.append(token)

&#x20;               elif not token.isalpha() and len(token) == 1:

&#x20;                   # Skip single punctuation

&#x20;                   removed\_tokens.append(token)

&#x20;               else:

&#x20;                   filtered\_tokens.append(token)

&#x20;           else:

&#x20;               removed\_tokens.append(token)

&#x20;       

&#x20;       print(f"  Removed ({len(removed\_tokens)}): {removed\_tokens}")

&#x20;       print(f"  Remaining ({len(filtered\_tokens)}): {filtered\_tokens}")

&#x20;       

&#x20;       return filtered\_tokens, removed\_tokens

&#x20;   

&#x20;   # =========================================

&#x20;   # STEP 4: STEMMING

&#x20;   # =========================================

&#x20;   def stem\_tokens(self, tokens):

&#x20;       """

&#x20;       Reduce words to their ROOT form by removing suffixes.

&#x20;       

&#x20;       Stemming is AGGRESSIVE — it chops off endings:

&#x20;       "running" → "run"

&#x20;       "organelles" → "organel"  (not a real word, but that's OK for matching)

&#x20;       "carrying" → "carri"

&#x20;       

&#x20;       Two algorithms:

&#x20;       - Porter Stemmer: More conservative, commonly used

&#x20;       - Lancaster Stemmer: More aggressive

&#x20;       

&#x20;       Why this matters: "cells", "cell", "cellular" all map to same root

&#x20;       → Better matching when a student asks about "cell" and textbook says "cells"

&#x20;       """

&#x20;       print("\\n" + "="\*60)

&#x20;       print("STEP 4: STEMMING")

&#x20;       print("="\*60)

&#x20;       

&#x20;       print(f"  Input tokens: {tokens}")

&#x20;       

&#x20;       porter\_stems = \[]

&#x20;       lancaster\_stems = \[]

&#x20;       

&#x20;       print(f"\\n  {'Token':<20} {'Porter Stem':<20} {'Lancaster Stem':<20}")

&#x20;       print(f"  {'-'\*20} {'-'\*20} {'-'\*20}")

&#x20;       

&#x20;       for token in tokens:

&#x20;           p\_stem = self.stemmer\_porter.stem(token)

&#x20;           l\_stem = self.stemmer\_lancaster.stem(token)

&#x20;           porter\_stems.append(p\_stem)

&#x20;           lancaster\_stems.append(l\_stem)

&#x20;           

&#x20;           # Highlight if stem differs from original

&#x20;           p\_marker = " ←" if p\_stem != token.lower() else ""

&#x20;           l\_marker = " ←" if l\_stem != token.lower() else ""

&#x20;           print(f"  {token:<20} {p\_stem:<20}{p\_marker:<5} {l\_stem:<20}{l\_marker}")

&#x20;       

&#x20;       print(f"\\n  → Using Porter Stemmer output: {porter\_stems}")

&#x20;       

&#x20;       return porter\_stems, lancaster\_stems

&#x20;   

&#x20;   # =========================================

&#x20;   # STEP 5: LEMMATIZATION

&#x20;   # =========================================

&#x20;   def lemmatize\_tokens(self, tokens):

&#x20;       """

&#x20;       Reduce words to their DICTIONARY form (lemma).

&#x20;       Unlike stemming, lemmatization produces REAL words:

&#x20;       

&#x20;       "running" → "run" (not "runn")

&#x20;       "organelles" → "organelle" (not "organel")

&#x20;       "better" → "good" (understands irregular forms!)

&#x20;       "carried" → "carry" (not "carri")

&#x20;       

&#x20;       Why this matters: Lemmas are real words that can be looked up.

&#x20;       Better for understanding meaning vs. just matching.

&#x20;       """

&#x20;       print("\\n" + "="\*60)

&#x20;       print("STEP 5: LEMMATIZATION")

&#x20;       print("="\*60)

&#x20;       

&#x20;       print(f"  Input tokens: {tokens}")

&#x20;       

&#x20;       # WordNet Lemmatizer needs POS tags for best results

&#x20;       # Default is noun; we'll try multiple POS

&#x20;       

&#x20;       lemmas = \[]

&#x20;       print(f"\\n  {'Token':<20} {'As Noun':<15} {'As Verb':<15} {'As Adj':<15} {'Best Lemma':<15}")

&#x20;       print(f"  {'-'\*20} {'-'\*15} {'-'\*15} {'-'\*15} {'-'\*15}")

&#x20;       

&#x20;       for token in tokens:

&#x20;           # Try all POS tags and pick the shortest result 

&#x20;           # (most reduced form)

&#x20;           noun\_lemma = self.lemmatizer.lemmatize(token.lower(), pos='n')

&#x20;           verb\_lemma = self.lemmatizer.lemmatize(token.lower(), pos='v')

&#x20;           adj\_lemma = self.lemmatizer.lemmatize(token.lower(), pos='a')

&#x20;           

&#x20;           # Pick the most reduced (shortest) real lemma

&#x20;           candidates = \[noun\_lemma, verb\_lemma, adj\_lemma]

&#x20;           best\_lemma = min(candidates, key=len)

&#x20;           

&#x20;           lemmas.append(best\_lemma)

&#x20;           

&#x20;           marker = " ←" if best\_lemma != token.lower() else ""

&#x20;           print(f"  {token:<20} {noun\_lemma:<15} {verb\_lemma:<15} {adj\_lemma:<15} {best\_lemma:<15}{marker}")

&#x20;       

&#x20;       print(f"\\n  → Lemmatized output: {lemmas}")

&#x20;       

&#x20;       return lemmas

&#x20;   

&#x20;   # =========================================

&#x20;   # STEP 6: POS TAGGING (Subject-Object-Word)

&#x20;   # =========================================

&#x20;   def pos\_tag(self, tokens):

&#x20;       """

&#x20;       Part-of-Speech tagging — identify the grammatical role of each word.

&#x20;       

&#x20;       This is what your mentor means by "Subject Object Word":

&#x20;       - Nouns (NN) = typically subjects/objects: "cell", "membrane"

&#x20;       - Verbs (VB) = actions: "contains", "performs"

&#x20;       - Adjectives (JJ) = descriptors: "fundamental", "large"

&#x20;       

&#x20;       Why this matters: 

&#x20;       1. Helps in query understanding (what is the student asking ABOUT?)

&#x20;       2. Essential for GEC module later

&#x20;       """

&#x20;       print("\\n" + "="\*60)

&#x20;       print("STEP 6: POS TAGGING (Subject-Object-Word Analysis)")

&#x20;       print("="\*60)

&#x20;       

&#x20;       print(f"  Input tokens: {tokens}")

&#x20;       

&#x20;       # NLTK POS tagging

&#x20;       pos\_tags = nltk.pos\_tag(tokens)

&#x20;       

&#x20;       print(f"\\n  {'Token':<20} {'POS Tag':<10} {'Meaning':<30}")

&#x20;       print(f"  {'-'\*20} {'-'\*10} {'-'\*30}")

&#x20;       

&#x20;       pos\_meanings = {

&#x20;           'NN': 'Noun (singular)',

&#x20;           'NNS': 'Noun (plural)',

&#x20;           'NNP': 'Proper Noun',

&#x20;           'VB': 'Verb (base form)',

&#x20;           'VBD': 'Verb (past tense)',

&#x20;           'VBG': 'Verb (gerund/-ing)',

&#x20;           'VBZ': 'Verb (3rd person)',

&#x20;           'VBP': 'Verb (non-3rd person)',

&#x20;           'JJ': 'Adjective',

&#x20;           'JJR': 'Adjective (comparative)',

&#x20;           'RB': 'Adverb',

&#x20;           'IN': 'Preposition',

&#x20;           'DT': 'Determiner',

&#x20;           'CC': 'Conjunction',

&#x20;           'PRP': 'Pronoun',

&#x20;           'MD': 'Modal verb',

&#x20;       }

&#x20;       

&#x20;       # Classify into subject, object, action categories

&#x20;       subjects = \[]  # Nouns that could be subjects

&#x20;       actions = \[]   # Verbs

&#x20;       descriptors = \[]  # Adjectives

&#x20;       

&#x20;       for word, tag in pos\_tags:

&#x20;           meaning = pos\_meanings.get(tag, f'Other ({tag})')

&#x20;           print(f"  {word:<20} {tag:<10} {meaning:<30}")

&#x20;           

&#x20;           if tag.startswith('NN'):

&#x20;               subjects.append(word)

&#x20;           elif tag.startswith('VB'):

&#x20;               actions.append(word)

&#x20;           elif tag.startswith('JJ'):

&#x20;               descriptors.append(word)

&#x20;       

&#x20;       print(f"\\n  Nouns (potential subjects/objects): {subjects}")

&#x20;       print(f"  Verbs (actions): {actions}")

&#x20;       print(f"  Adjectives (descriptors): {descriptors}")

&#x20;       

&#x20;       return pos\_tags, {

&#x20;           'nouns': subjects,

&#x20;           'verbs': actions,

&#x20;           'adjectives': descriptors

&#x20;       }

&#x20;   

&#x20;   # =========================================

&#x20;   # COMPLETE PIPELINE — ALL STEPS TOGETHER

&#x20;   # =========================================

&#x20;   def process\_sentence(self, sentence, show\_steps=True):

&#x20;       """

&#x20;       Run the COMPLETE preprocessing pipeline on ONE sentence.

&#x20;       Returns a dictionary with output from every step.

&#x20;       """

&#x20;       print("\\n" + "#"\*70)

&#x20;       print(f"# PROCESSING: \\"{sentence\[:70]}{'...' if len(sentence)>70 else ''}\\"")

&#x20;       print("#"\*70)

&#x20;       

&#x20;       result = {

&#x20;           "original": sentence,

&#x20;           "steps": {}

&#x20;       }

&#x20;       

&#x20;       # Step 2: Tokenization

&#x20;       tokens = self.tokenize(sentence)

&#x20;       result\["steps"]\["tokenization"] = tokens

&#x20;       

&#x20;       # Step 3: Stop word removal

&#x20;       filtered\_tokens, removed = self.remove\_stop\_words(tokens)

&#x20;       result\["steps"]\["stop\_word\_removal"] = {

&#x20;           "kept": filtered\_tokens,

&#x20;           "removed": removed

&#x20;       }

&#x20;       

&#x20;       # Step 4: Stemming

&#x20;       porter\_stems, lancaster\_stems = self.stem\_tokens(filtered\_tokens)

&#x20;       result\["steps"]\["stemming"] = {

&#x20;           "porter": porter\_stems,

&#x20;           "lancaster": lancaster\_stems

&#x20;       }

&#x20;       

&#x20;       # Step 5: Lemmatization

&#x20;       lemmas = self.lemmatize\_tokens(filtered\_tokens)

&#x20;       result\["steps"]\["lemmatization"] = lemmas

&#x20;       

&#x20;       # Step 6: POS Tagging (on original filtered tokens, not stems)

&#x20;       pos\_tags, categories = self.pos\_tag(filtered\_tokens)

&#x20;       result\["steps"]\["pos\_tagging"] = {

&#x20;           "tags": \[(w, t) for w, t in pos\_tags],

&#x20;           "categories": categories

&#x20;       }

&#x20;       

&#x20;       return result

&#x20;   

&#x20;   def process\_text(self, text, max\_sentences=None):

&#x20;       """

&#x20;       Run complete pipeline on full text.

&#x20;       """

&#x20;       # Step 1: Sentence Segmentation

&#x20;       sentences = self.segment\_sentences(text)

&#x20;       

&#x20;       if max\_sentences:

&#x20;           sentences = sentences\[:max\_sentences]

&#x20;           print(f"\\n\[INFO] Processing first {max\_sentences} sentences (as per mentor's instruction)")

&#x20;       

&#x20;       all\_results = \[]

&#x20;       for i, sentence in enumerate(sentences, 1):

&#x20;           print(f"\\n\\n{'\*'\*70}")

&#x20;           print(f"\* SENTENCE {i}/{len(sentences)}")

&#x20;           print(f"{'\*'\*70}")

&#x20;           

&#x20;           result = self.process\_sentence(sentence)

&#x20;           result\["sentence\_number"] = i

&#x20;           all\_results.append(result)

&#x20;       

&#x20;       return all\_results

&#x20;   

&#x20;   def save\_results(self, results, output\_path):

&#x20;       """Save pipeline results as JSON."""

&#x20;       # Convert tuples to lists for JSON serialization

&#x20;       serializable = json.loads(

&#x20;           json.dumps(results, default=str, ensure\_ascii=False)

&#x20;       )

&#x20;       

&#x20;       os.makedirs(os.path.dirname(output\_path), exist\_ok=True)

&#x20;       with open(output\_path, 'w', encoding='utf-8') as f:

&#x20;           json.dump(serializable, f, indent=2, ensure\_ascii=False)

&#x20;       print(f"\\n\[SAVED] Results → {output\_path}")





\# ============================================

\# USAGE — Run the preprocessing pipeline

\# ============================================

if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   import os

&#x20;   

&#x20;   # Initialize preprocessor

&#x20;   preprocessor = EnglishPreprocessor()

&#x20;   

&#x20;   # ---- DEMO ON 2 SENTENCES (as mentor said) ----

&#x20;   demo\_sentences = \[

&#x20;       "All living organisms are made up of cells, which are the fundamental structural and functional units of life.",

&#x20;       "The cell membrane is a selectively permeable membrane that controls the movement of substances into and out of the cell."

&#x20;   ]

&#x20;   

&#x20;   print("\\n" + "="\*70)

&#x20;   print("DEMO: Processing 2 Biology sentences from NCERT Chapter 5")

&#x20;   print("="\*70)

&#x20;   

&#x20;   for sent in demo\_sentences:

&#x20;       result = preprocessor.process\_sentence(sent)

&#x20;   

&#x20;   # ---- ON FULL CHAPTER (after demo works) ----

&#x20;   # Uncomment this when ready:

&#x20;   """

&#x20;   with open("../data/processed/chapter\_text\_clean.txt", 'r') as f:

&#x20;       chapter\_text = f.read()

&#x20;   

&#x20;   results = preprocessor.process\_text(chapter\_text, max\_sentences=5)

&#x20;   preprocessor.save\_results(results, "../outputs/pipeline\_results/english\_preprocessing.json")

&#x20;   """

```



\---



\## STEP 3: HINGLISH DATASET CREATION (Jatin — Task J1)



This is the most important creative task. You need to convert English NCERT sentences into natural Hinglish.



\### File: `src/hinglish\_dataset\_creator.py`



```python

"""

EduHinglish - Hinglish Dataset Creator

Author: Jatin

Purpose: Create code-mixed (Hinglish) versions of NCERT Biology sentences

&#x20;        for preprocessing pipeline testing



THIS IS THE LABELED DATASET YOUR MENTOR ASKED FOR.

"""



import json

import os





class HinglishDatasetCreator:

&#x20;   """

&#x20;   Create and manage Hinglish educational sentences dataset.

&#x20;   

&#x20;   Each entry has:

&#x20;   - original\_english: The NCERT English sentence

&#x20;   - hinglish\_roman: Hinglish version in Roman script

&#x20;   - hinglish\_devanagari: Hinglish version in Devanagari script (partial)

&#x20;   - hinglish\_mixed: Mixed script version

&#x20;   - word\_level\_labels: Language tag for each word \[HI/EN/NE/UNIV]

&#x20;   - topic: The biology topic

&#x20;   - chapter: Chapter reference

&#x20;   """

&#x20;   

&#x20;   def \_\_init\_\_(self):

&#x20;       self.dataset = \[]

&#x20;       print("\[OK] Hinglish Dataset Creator initialized")

&#x20;   

&#x20;   def create\_biology\_chapter5\_dataset(self):

&#x20;       """

&#x20;       Create Hinglish sentences for NCERT Class 9 Biology Chapter 5:

&#x20;       'The Fundamental Unit of Life'

&#x20;       

&#x20;       RULES for natural Hinglish:

&#x20;       1. Keep technical/scientific terms in English 

&#x20;          (cell, membrane, nucleus, organelle, etc.)

&#x20;       2. Use Hindi for:

&#x20;          - Connectors: "aur", "ya", "lekin", "kyunki", "isliye"

&#x20;          - Verbs: "hota hai", "karta hai", "banta hai"

&#x20;          - Explanatory phrases: "matlab", "yaani", "iska matlab"

&#x20;       3. Keep it natural — how a PW/Unacademy teacher would actually say it

&#x20;       """

&#x20;       

&#x20;       self.dataset = \[

&#x20;           # ================================================

&#x20;           # SENTENCE 1: What is a cell?

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 1,

&#x20;               "original\_english": "All living organisms are made up of cells.",

&#x20;               "hinglish\_roman": "Sabhi living organisms cells se bane hote hain.",

&#x20;               "hinglish\_devanagari": "सभी living organisms cells से बने होते हैं।",

&#x20;               "hinglish\_mixed": "Sabhi living organisms cells से bane hote hैं.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Sabhi": "HI",

&#x20;                   "living": "EN",

&#x20;                   "organisms": "EN",

&#x20;                   "cells": "EN",

&#x20;                   "se": "HI",

&#x20;                   "bane": "HI",

&#x20;                   "hote": "HI",

&#x20;                   "hain": "HI"

&#x20;               },

&#x20;               "topic": "Introduction to Cell",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "inter-sentential",

&#x20;               "notes": "Technical terms (living, organisms, cells) kept in English; Hindi used for grammar structure"

&#x20;           },

&#x20;           

&#x20;           # ================================================

&#x20;           # SENTENCE 2: Cell as fundamental unit

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 2,

&#x20;               "original\_english": "The cell is the fundamental structural and functional unit of life.",

&#x20;               "hinglish\_roman": "Cell life ki fundamental structural aur functional unit hai.",

&#x20;               "hinglish\_devanagari": "Cell life की fundamental structural और functional unit है।",

&#x20;               "hinglish\_mixed": "Cell life ki fundamental structural aur functional unit है.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Cell": "EN",

&#x20;                   "life": "EN",

&#x20;                   "ki": "HI",

&#x20;                   "fundamental": "EN",

&#x20;                   "structural": "EN",

&#x20;                   "aur": "HI",

&#x20;                   "functional": "EN",

&#x20;                   "unit": "EN",

&#x20;                   "hai": "HI"

&#x20;               },

&#x20;               "topic": "Cell Theory",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Most words are English technical terms; Hindi provides grammatical glue"

&#x20;           },

&#x20;           

&#x20;           # ================================================

&#x20;           # SENTENCE 3: Cell membrane

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 3,

&#x20;               "original\_english": "The cell membrane is a selectively permeable membrane that controls the movement of substances into and out of the cell.",

&#x20;               "hinglish\_roman": "Cell membrane ek selectively permeable membrane hoti hai jo substances ke movement ko cell ke andar aur bahar control karti hai.",

&#x20;               "hinglish\_devanagari": "Cell membrane एक selectively permeable membrane होती है जो substances के movement को cell के अंदर और बाहर control करती है।",

&#x20;               "hinglish\_mixed": "Cell membrane एक selectively permeable membrane hoti hai jo substances ke movement ko cell ke andar aur bahar control karti hai.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Cell": "EN",

&#x20;                   "membrane": "EN",

&#x20;                   "ek": "HI",

&#x20;                   "selectively": "EN",

&#x20;                   "permeable": "EN",

&#x20;                   "hoti": "HI",

&#x20;                   "hai": "HI",

&#x20;                   "jo": "HI",

&#x20;                   "substances": "EN",

&#x20;                   "ke": "HI",

&#x20;                   "movement": "EN",

&#x20;                   "ko": "HI",

&#x20;                   "andar": "HI",

&#x20;                   "aur": "HI",

&#x20;                   "bahar": "HI",

&#x20;                   "control": "EN",

&#x20;                   "karti": "HI"

&#x20;               },

&#x20;               "topic": "Cell Membrane / Plasma Membrane",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Dense code-mixing — Hindi structural words woven between English technical terms"

&#x20;           },

&#x20;           

&#x20;           # ================================================

&#x20;           # SENTENCE 4: Nucleus

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 4,

&#x20;               "original\_english": "The nucleus contains chromosomes which carry genes and help in inheritance or transfer of characters from parents to offspring.",

&#x20;               "hinglish\_roman": "Nucleus mein chromosomes hote hain jinmein genes hote hain aur yeh inheritance ya characters ke transfer mein parents se offspring tak help karte hain.",

&#x20;               "hinglish\_devanagari": "Nucleus में chromosomes होते हैं जिनमें genes होते हैं और यह inheritance या characters के transfer में parents से offspring तक help करते हैं।",

&#x20;               "hinglish\_mixed": "Nucleus mein chromosomes hote hain jinmein genes hote hain aur yeh inheritance ya characters ke transfer में parents se offspring tak help karte hain.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Nucleus": "EN",

&#x20;                   "mein": "HI",

&#x20;                   "chromosomes": "EN",

&#x20;                   "hote": "HI",

&#x20;                   "hain": "HI",

&#x20;                   "jinmein": "HI",

&#x20;                   "genes": "EN",

&#x20;                   "aur": "HI",

&#x20;                   "yeh": "HI",

&#x20;                   "inheritance": "EN",

&#x20;                   "ya": "HI",

&#x20;                   "characters": "EN",

&#x20;                   "ke": "HI",

&#x20;                   "transfer": "EN",

&#x20;                   "parents": "EN",

&#x20;                   "se": "HI",

&#x20;                   "offspring": "EN",

&#x20;                   "tak": "HI",

&#x20;                   "help": "EN",

&#x20;                   "karte": "HI"

&#x20;               },

&#x20;               "topic": "Nucleus",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Heavy code-mixing with biology-specific terms"

&#x20;           },

&#x20;           

&#x20;           # ================================================

&#x20;           # SENTENCE 5: Osmosis

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 5,

&#x20;               "original\_english": "Osmosis is the movement of water molecules through a selectively permeable membrane from a region of high water concentration to a region of low water concentration.",

&#x20;               "hinglish\_roman": "Osmosis mein water molecules selectively permeable membrane se hokar high concentration wale region se low concentration wale region ki taraf move karte hain.",

&#x20;               "hinglish\_devanagari": "Osmosis में water molecules selectively permeable membrane से होकर high concentration वाले region से low concentration वाले region की तरफ move करते हैं।",

&#x20;               "hinglish\_mixed": "Osmosis mein water molecules selectively permeable membrane se hokar high concentration wale region se low concentration wale region ki taraf move karte hain.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Osmosis": "EN",

&#x20;                   "mein": "HI",

&#x20;                   "water": "EN",

&#x20;                   "molecules": "EN",

&#x20;                   "selectively": "EN",

&#x20;                   "permeable": "EN",

&#x20;                   "membrane": "EN",

&#x20;                   "se": "HI",

&#x20;                   "hokar": "HI",

&#x20;                   "high": "EN",

&#x20;                   "concentration": "EN",

&#x20;                   "wale": "HI",

&#x20;                   "region": "EN",

&#x20;                   "low": "EN",

&#x20;                   "ki": "HI",

&#x20;                   "taraf": "HI",

&#x20;                   "move": "EN",

&#x20;                   "karte": "HI",

&#x20;                   "hain": "HI"

&#x20;               },

&#x20;               "topic": "Osmosis",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Scientific process description in Hinglish"

&#x20;           },

&#x20;           

&#x20;           # ================================================

&#x20;           # SENTENCE 6: Mitochondria (Student-style question)

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 6,

&#x20;               "original\_english": "Mitochondria are known as the powerhouse of the cell because they produce energy in the form of ATP.",

&#x20;               "hinglish\_roman": "Mitochondria ko cell ka powerhouse kaha jaata hai kyunki yeh ATP ke form mein energy produce karte hain.",

&#x20;               "hinglish\_devanagari": "Mitochondria को cell का powerhouse कहा जाता है क्योंकि यह ATP के form में energy produce करते हैं।",

&#x20;               "hinglish\_mixed": "Mitochondria को cell ka powerhouse kaha jaata hai क्योंकि yeh ATP ke form mein energy produce karte hain.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Mitochondria": "EN",

&#x20;                   "ko": "HI",

&#x20;                   "cell": "EN",

&#x20;                   "ka": "HI",

&#x20;                   "powerhouse": "EN",

&#x20;                   "kaha": "HI",

&#x20;                   "jaata": "HI",

&#x20;                   "hai": "HI",

&#x20;                   "kyunki": "HI",

&#x20;                   "yeh": "HI",

&#x20;                   "ATP": "EN",

&#x20;                   "ke": "HI",

&#x20;                   "form": "EN",

&#x20;                   "mein": "HI",

&#x20;                   "energy": "EN",

&#x20;                   "produce": "EN",

&#x20;                   "karte": "HI",

&#x20;                   "hain": "HI"

&#x20;               },

&#x20;               "topic": "Mitochondria",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Classic biology fact expressed in natural Hinglish"

&#x20;           },

&#x20;           

&#x20;           # ================================================

&#x20;           # SENTENCE 7: Prokaryotic vs Eukaryotic

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 7,

&#x20;               "original\_english": "Prokaryotic cells do not have a well-defined nucleus, while eukaryotic cells have a well-organised nucleus with a nuclear membrane.",

&#x20;               "hinglish\_roman": "Prokaryotic cells mein well-defined nucleus nahi hota, jabki eukaryotic cells mein nuclear membrane ke saath ek well-organised nucleus hota hai.",

&#x20;               "hinglish\_devanagari": "Prokaryotic cells में well-defined nucleus नहीं होता, जबकि eukaryotic cells में nuclear membrane के साथ एक well-organised nucleus होता है।",

&#x20;               "hinglish\_mixed": "Prokaryotic cells mein well-defined nucleus nahi hota, jabki eukaryotic cells mein nuclear membrane ke saath ek well-organised nucleus hota hai.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Prokaryotic": "EN",

&#x20;                   "cells": "EN",

&#x20;                   "mein": "HI",

&#x20;                   "well-defined": "EN",

&#x20;                   "nucleus": "EN",

&#x20;                   "nahi": "HI",

&#x20;                   "hota": "HI",

&#x20;                   "jabki": "HI",

&#x20;                   "eukaryotic": "EN",

&#x20;                   "nuclear": "EN",

&#x20;                   "membrane": "EN",

&#x20;                   "ke": "HI",

&#x20;                   "saath": "HI",

&#x20;                   "ek": "HI",

&#x20;                   "well-organised": "EN",

&#x20;                   "hai": "HI"

&#x20;               },

&#x20;               "topic": "Prokaryotic and Eukaryotic Cells",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Comparison-type sentence — common in biology"

&#x20;           },

&#x20;           

&#x20;           # ================================================

&#x20;           # SENTENCE 8: Student QUESTION style (Doubt)

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 8,

&#x20;               "original\_english": "What is the function of the endoplasmic reticulum in the cell?",

&#x20;               "hinglish\_roman": "Sir, endoplasmic reticulum ka cell mein kya function hota hai?",

&#x20;               "hinglish\_devanagari": "Sir, endoplasmic reticulum का cell में क्या function होता है?",

&#x20;               "hinglish\_mixed": "Sir, endoplasmic reticulum ka cell में kya function hota hai?",

&#x20;               "word\_level\_labels": {

&#x20;                   "Sir": "UNIV",

&#x20;                   "endoplasmic": "EN",

&#x20;                   "reticulum": "EN",

&#x20;                   "ka": "HI",

&#x20;                   "cell": "EN",

&#x20;                   "mein": "HI",

&#x20;                   "kya": "HI",

&#x20;                   "function": "EN",

&#x20;                   "hota": "HI",

&#x20;                   "hai": "HI"

&#x20;               },

&#x20;               "topic": "Endoplasmic Reticulum",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "is\_student\_query": True,

&#x20;               "intent": "explain\_concept",

&#x20;               "notes": "Natural student doubt in Hinglish — this is the INPUT our system will receive"

&#x20;           },

&#x20;           

&#x20;           # ================================================

&#x20;           # SENTENCE 9: Student question with spelling variation

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 9,

&#x20;               "original\_english": "Why are lysosomes called suicide bags?",

&#x20;               "hinglish\_roman": "Lysosomes ko suicide bags kyun kehte hain?",

&#x20;               "hinglish\_devanagari": "Lysosomes को suicide bags क्यों कहते हैं?",

&#x20;               "hinglish\_mixed": "Lysosomes ko suicide bags kyun kehte hैं?",

&#x20;               "word\_level\_labels": {

&#x20;                   "Lysosomes": "EN",

&#x20;                   "ko": "HI",

&#x20;                   "suicide": "EN",

&#x20;                   "bags": "EN",

&#x20;                   "kyun": "HI",

&#x20;                   "kehte": "HI",

&#x20;                   "hain": "HI"

&#x20;               },

&#x20;               "topic": "Lysosomes",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "is\_student\_query": True,

&#x20;               "intent": "explain\_concept",

&#x20;               "notes": "Short, direct student question"

&#x20;           },

&#x20;           

&#x20;           # ================================================

&#x20;           # SENTENCE 10: With grammatical error (for GEC testing)

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 10,

&#x20;               "original\_english": "The cell wall is present in plant cells but not in animal cells.",

&#x20;               "hinglish\_roman": "Cell wall plant cells mein hota hai lekin animal cells mein nahi hota.",

&#x20;               "hinglish\_with\_error": "Cell wall plant cells mein hota hai lekin animal cells mein nahi hote.",

&#x20;               "error\_type": "subject-verb agreement",

&#x20;               "error\_explanation": "'hote' should be 'hota' because 'cell wall' is singular",

&#x20;               "word\_level\_labels": {

&#x20;                   "Cell": "EN",

&#x20;                   "wall": "EN",

&#x20;                   "plant": "EN",

&#x20;                   "cells": "EN",

&#x20;                   "mein": "HI",

&#x20;                   "hota": "HI",

&#x20;                   "hai": "HI",

&#x20;                   "lekin": "HI",

&#x20;                   "animal": "EN",

&#x20;                   "nahi": "HI"

&#x20;               },

&#x20;               "topic": "Cell Wall",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "is\_gec\_sample": True,

&#x20;               "notes": "Sample with intentional error for GEC module testing"

&#x20;           },



&#x20;           # ================================================

&#x20;           # SENTENCES 11-15: More variety

&#x20;           # ================================================

&#x20;           {

&#x20;               "id": 11,

&#x20;               "original\_english": "Robert Hooke discovered cells in 1665 by observing a thin slice of cork under a microscope.",

&#x20;               "hinglish\_roman": "Robert Hooke ne 1665 mein cork ki thin slice ko microscope se observe karke cells discover kiya tha.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Robert": "NE", "Hooke": "NE", "ne": "HI", "1665": "UNIV",

&#x20;                   "mein": "HI", "cork": "EN", "ki": "HI", "thin": "EN",

&#x20;                   "slice": "EN", "ko": "HI", "microscope": "EN", "se": "HI",

&#x20;                   "observe": "EN", "karke": "HI", "cells": "EN",

&#x20;                   "discover": "EN", "kiya": "HI", "tha": "HI"

&#x20;               },

&#x20;               "topic": "Discovery of Cell",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Historical fact with named entity (Robert Hooke)"

&#x20;           },

&#x20;           {

&#x20;               "id": 12,

&#x20;               "original\_english": "Plastids are present only in plant cells and they contain pigments like chlorophyll.",

&#x20;               "hinglish\_roman": "Plastids sirf plant cells mein hote hain aur inmein chlorophyll jaise pigments hote hain.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Plastids": "EN", "sirf": "HI", "plant": "EN", "cells": "EN",

&#x20;                   "mein": "HI", "hote": "HI", "hain": "HI", "aur": "HI",

&#x20;                   "inmein": "HI", "chlorophyll": "EN", "jaise": "HI",

&#x20;                   "pigments": "EN"

&#x20;               },

&#x20;               "topic": "Plastids",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Simple fact with code-mixing"

&#x20;           },

&#x20;           {

&#x20;               "id": 13,

&#x20;               "original\_english": "Can you explain the difference between diffusion and osmosis?",

&#x20;               "hinglish\_roman": "Sir diffusion aur osmosis mein kya difference hai samjha dijiye?",

&#x20;               "word\_level\_labels": {

&#x20;                   "Sir": "UNIV", "diffusion": "EN", "aur": "HI", "osmosis": "EN",

&#x20;                   "mein": "HI", "kya": "HI", "difference": "EN", "hai": "HI",

&#x20;                   "samjha": "HI", "dijiye": "HI"

&#x20;               },

&#x20;               "topic": "Diffusion and Osmosis",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "is\_student\_query": True,

&#x20;               "intent": "compare\_concepts",

&#x20;               "notes": "Comparison question from student"

&#x20;           },

&#x20;           {

&#x20;               "id": 14,

&#x20;               "original\_english": "The Golgi apparatus is responsible for packaging and dispatching materials within the cell.",

&#x20;               "hinglish\_roman": "Golgi apparatus cell ke andar materials ko package aur dispatch karne ka kaam karta hai.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Golgi": "EN", "apparatus": "EN", "cell": "EN", "ke": "HI",

&#x20;                   "andar": "HI", "materials": "EN", "ko": "HI", "package": "EN",

&#x20;                   "aur": "HI", "dispatch": "EN", "karne": "HI", "ka": "HI",

&#x20;                   "kaam": "HI", "karta": "HI", "hai": "HI"

&#x20;               },

&#x20;               "topic": "Golgi Apparatus",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Function description in Hinglish"

&#x20;           },

&#x20;           {

&#x20;               "id": 15,

&#x20;               "original\_english": "Vacuoles in plant cells are large and provide turgidity and rigidity to the cell.",

&#x20;               "hinglish\_roman": "Plant cells mein vacuoles bade hote hain aur yeh cell ko turgidity aur rigidity provide karte hain.",

&#x20;               "word\_level\_labels": {

&#x20;                   "Plant": "EN", "cells": "EN", "mein": "HI", "vacuoles": "EN",

&#x20;                   "bade": "HI", "hote": "HI", "hain": "HI", "aur": "HI",

&#x20;                   "yeh": "HI", "cell": "EN", "ko": "HI", "turgidity": "EN",

&#x20;                   "rigidity": "EN", "provide": "EN", "karte": "HI"

&#x20;               },

&#x20;               "topic": "Vacuoles",

&#x20;               "chapter": "Chapter 5: The Fundamental Unit of Life",

&#x20;               "class": "9",

&#x20;               "code\_mixing\_type": "intra-sentential",

&#x20;               "notes": "Biology fact with technical terms preserved in English"

&#x20;           }

&#x20;       ]

&#x20;       

&#x20;       print(f"\[OK] Created {len(self.dataset)} Hinglish sentences for Biology Chapter 5")

&#x20;       return self.dataset

&#x20;   

&#x20;   def get\_statistics(self):

&#x20;       """Calculate dataset statistics."""

&#x20;       total = len(self.dataset)

&#x20;       queries = sum(1 for d in self.dataset if d.get('is\_student\_query', False))

&#x20;       gec\_samples = sum(1 for d in self.dataset if d.get('is\_gec\_sample', False))

&#x20;       statements = total - queries - gec\_samples

&#x20;       

&#x20;       # Calculate average Code-Mixing Index

&#x20;       cmis = \[]

&#x20;       for entry in self.dataset:

&#x20;           labels = entry.get('word\_level\_labels', {})

&#x20;           if labels:

&#x20;               hi\_count = sum(1 for v in labels.values() if v == 'HI')

&#x20;               en\_count = sum(1 for v in labels.values() if v == 'EN')

&#x20;               total\_words = hi\_count + en\_count

&#x20;               if total\_words > 0:

&#x20;                   cmi = (1 - max(hi\_count, en\_count) / total\_words) \* 100

&#x20;                   cmis.append(cmi)

&#x20;       

&#x20;       avg\_cmi = sum(cmis) / len(cmis) if cmis else 0

&#x20;       

&#x20;       # Collect all topics

&#x20;       topics = list(set(d\['topic'] for d in self.dataset))

&#x20;       

&#x20;       stats = {

&#x20;           "total\_sentences": total,

&#x20;           "statement\_sentences": statements,

&#x20;           "student\_queries": queries,

&#x20;           "gec\_samples": gec\_samples,

&#x20;           "average\_cmi": round(avg\_cmi, 2),

&#x20;           "topics\_covered": topics,

&#x20;           "num\_topics": len(topics)

&#x20;       }

&#x20;       

&#x20;       print("\\n" + "="\*50)

&#x20;       print("DATASET STATISTICS")

&#x20;       print("="\*50)

&#x20;       for key, value in stats.items():

&#x20;           if isinstance(value, list):

&#x20;               print(f"  {key}:")

&#x20;               for item in value:

&#x20;                   print(f"    - {item}")

&#x20;           else:

&#x20;               print(f"  {key}: {value}")

&#x20;       print("="\*50)

&#x20;       

&#x20;       return stats

&#x20;   

&#x20;   def save\_dataset(self, output\_path):

&#x20;       """Save dataset as JSON."""

&#x20;       os.makedirs(os.path.dirname(output\_path), exist\_ok=True)

&#x20;       with open(output\_path, 'w', encoding='utf-8') as f:

&#x20;           json.dump(self.dataset, f, indent=2, ensure\_ascii=False)

&#x20;       print(f"\[SAVED] Hinglish dataset → {output\_path}")





\# ============================================

\# USAGE

\# ============================================

if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   creator = HinglishDatasetCreator()

&#x20;   dataset = creator.create\_biology\_chapter5\_dataset()

&#x20;   creator.get\_statistics()

&#x20;   creator.save\_dataset("../data/hinglish/hinglish\_bio\_ch5.json")

&#x20;   

&#x20;   # Preview

&#x20;   print("\\n\[PREVIEW] First 3 entries:")

&#x20;   for entry in dataset\[:3]:

&#x20;       print(f"\\n  ID: {entry\['id']}")

&#x20;       print(f"  English:  {entry\['original\_english']}")

&#x20;       print(f"  Hinglish: {entry\['hinglish\_roman']}")

&#x20;       print(f"  Labels:   {entry\['word\_level\_labels']}")

&#x20;       print(f"  Topic:    {entry\['topic']}")

```



\---



\## STEP 4: SCRIPT DETECTION \& LANGUAGE IDENTIFICATION (Jatin — Tasks J2, J3, J4)



\### File: `src/script\_detector.py`



```python

"""

EduHinglish - Script Detection, Normalization \& Language Identification

Author: Jatin

Purpose: Detect script (Roman/Devanagari/Mixed), normalize text, 

&#x20;        and identify language at word level

"""



import re

import unicodedata

from collections import Counter





class ScriptDetector:

&#x20;   """

&#x20;   Detect and classify the script of input text.

&#x20;   Handles: Roman (Latin), Devanagari, Mixed

&#x20;   """

&#x20;   

&#x20;   # Unicode ranges

&#x20;   DEVANAGARI\_RANGE = (0x0900, 0x097F)

&#x20;   LATIN\_RANGE\_UPPER = (0x0041, 0x005A)

&#x20;   LATIN\_RANGE\_LOWER = (0x0061, 0x007A)

&#x20;   

&#x20;   def detect\_char\_script(self, char):

&#x20;       """Detect script of a single character."""

&#x20;       code\_point = ord(char)

&#x20;       

&#x20;       if self.DEVANAGARI\_RANGE\[0] <= code\_point <= self.DEVANAGARI\_RANGE\[1]:

&#x20;           return "DEVANAGARI"

&#x20;       elif (self.LATIN\_RANGE\_UPPER\[0] <= code\_point <= self.LATIN\_RANGE\_UPPER\[1] or

&#x20;             self.LATIN\_RANGE\_LOWER\[0] <= code\_point <= self.LATIN\_RANGE\_LOWER\[1]):

&#x20;           return "ROMAN"

&#x20;       elif char.isdigit():

&#x20;           return "NUMERIC"

&#x20;       elif char.isspace():

&#x20;           return "SPACE"

&#x20;       else:

&#x20;           return "OTHER"

&#x20;   

&#x20;   def detect\_word\_script(self, word):

&#x20;       """Detect the dominant script of a word."""

&#x20;       scripts = Counter()

&#x20;       

&#x20;       for char in word:

&#x20;           script = self.detect\_char\_script(char)

&#x20;           if script not in ("SPACE", "OTHER", "NUMERIC"):

&#x20;               scripts\[script] += 1

&#x20;       

&#x20;       if not scripts:

&#x20;           return "OTHER"

&#x20;       

&#x20;       total\_alpha = sum(scripts.values())

&#x20;       

&#x20;       if scripts.get("DEVANAGARI", 0) > 0 and scripts.get("ROMAN", 0) > 0:

&#x20;           return "MIXED"

&#x20;       elif scripts.get("DEVANAGARI", 0) > 0:

&#x20;           return "DEVANAGARI"

&#x20;       elif scripts.get("ROMAN", 0) > 0:

&#x20;           return "ROMAN"

&#x20;       else:

&#x20;           return "OTHER"

&#x20;   

&#x20;   def detect\_text\_script(self, text):

&#x20;       """

&#x20;       Detect the overall script composition of input text.

&#x20;       Returns: script type and detailed breakdown.

&#x20;       """

&#x20;       print("\\n" + "="\*60)

&#x20;       print("SCRIPT DETECTION")

&#x20;       print("="\*60)

&#x20;       print(f"  Input: \\"{text}\\"")

&#x20;       

&#x20;       # Analyze each word

&#x20;       words = text.split()

&#x20;       word\_scripts = {}

&#x20;       

&#x20;       for word in words:

&#x20;           # Remove punctuation for script analysis

&#x20;           clean\_word = re.sub(r'\[^\\w]', '', word)

&#x20;           if clean\_word:

&#x20;               word\_scripts\[word] = self.detect\_word\_script(clean\_word)

&#x20;       

&#x20;       # Count script distribution

&#x20;       script\_counts = Counter(word\_scripts.values())

&#x20;       total\_words = len(word\_scripts)

&#x20;       

&#x20;       # Determine overall script

&#x20;       roman\_pct = script\_counts.get("ROMAN", 0) / max(total\_words, 1) \* 100

&#x20;       dev\_pct = script\_counts.get("DEVANAGARI", 0) / max(total\_words, 1) \* 100

&#x20;       mixed\_pct = script\_counts.get("MIXED", 0) / max(total\_words, 1) \* 100

&#x20;       

&#x20;       if dev\_pct == 0 and mixed\_pct == 0:

&#x20;           overall\_script = "ROMAN"

&#x20;       elif roman\_pct == 0 and mixed\_pct == 0:

&#x20;           overall\_script = "DEVANAGARI"

&#x20;       else:

&#x20;           overall\_script = "MIXED"

&#x20;       

&#x20;       # Print results

&#x20;       print(f"\\n  Word-level script analysis:")

&#x20;       print(f"  {'Word':<25} {'Script':<15}")

&#x20;       print(f"  {'-'\*25} {'-'\*15}")

&#x20;       for word, script in word\_scripts.items():

&#x20;           print(f"  {word:<25} {script:<15}")

&#x20;       

&#x20;       print(f"\\n  Script Distribution:")

&#x20;       print(f"    Roman:      {script\_counts.get('ROMAN', 0)} words ({roman\_pct:.1f}%)")

&#x20;       print(f"    Devanagari: {script\_counts.get('DEVANAGARI', 0)} words ({dev\_pct:.1f}%)")

&#x20;       print(f"    Mixed:      {script\_counts.get('MIXED', 0)} words ({mixed\_pct:.1f}%)")

&#x20;       

&#x20;       print(f"\\n  → Overall Script: {overall\_script}")

&#x20;       

&#x20;       return {

&#x20;           "overall\_script": overall\_script,

&#x20;           "word\_scripts": word\_scripts,

&#x20;           "distribution": {

&#x20;               "roman\_pct": round(roman\_pct, 1),

&#x20;               "devanagari\_pct": round(dev\_pct, 1),

&#x20;               "mixed\_pct": round(mixed\_pct, 1)

&#x20;           }

&#x20;       }





class HinglishNormalizer:

&#x20;   """

&#x20;   Normalize Hinglish text:

&#x20;   1. Spelling variations → standard form

&#x20;   2. Common abbreviations → full form

&#x20;   3. Script unification (optional)

&#x20;   """

&#x20;   

&#x20;   def \_\_init\_\_(self):

&#x20;       # Common Hinglish spelling variations mapping

&#x20;       self.spelling\_map = {

&#x20;           # Variants of common Hindi words in Roman script

&#x20;           "kya": \["kia", "kyaa", "keya", "kiya"],

&#x20;           "hai": \["he", "h", "hae", "hay"],

&#x20;           "hain": \["hein", "hn", "haen"],

&#x20;           "nahi": \["nhi", "nahin", "nai", "ni"],

&#x20;           "mein": \["me", "mei", "main", "mn"],

&#x20;           "aur": \["or", "our", "ar", "aor"],

&#x20;           "yeh": \["ye", "yah", "y"],

&#x20;           "kaise": \["kese", "kaise", "kse"],

&#x20;           "kyun": \["kyu", "kyunki", "kyon", "q"],

&#x20;           "samjhao": \["samjhao", "smjhao", "samjao", "smjao"],

&#x20;           "batao": \["btao", "batao", "btaao"],

&#x20;           "karo": \["kro", "karo", "krdo"],

&#x20;           "hota": \["hota", "hta", "hoti"],

&#x20;           "karta": \["krta", "karta", "krti"],

&#x20;           "wala": \["wale", "wali", "waala", "waale"],

&#x20;           "achha": \["acha", "achha", "accha", "achchha"],

&#x20;           "bahut": \["bhut", "bahot", "bohot", "boht"],

&#x20;           "lekin": \["lekn", "lekin", "lkin", "par"],

&#x20;           "iska": \["iska", "ika", "uska"],

&#x20;           "sabhi": \["sabhi", "sab", "sbhi"],

&#x20;           "dijiye": \["dijiye", "dijie", "do", "dena"],

&#x20;       }

&#x20;       

&#x20;       # Build reverse lookup

&#x20;       self.reverse\_map = {}

&#x20;       for standard, variants in self.spelling\_map.items():

&#x20;           for variant in variants:

&#x20;               if variant != standard:

&#x20;                   self.reverse\_map\[variant.lower()] = standard

&#x20;       

&#x20;       # Common texting abbreviations

&#x20;       self.abbreviations = {

&#x20;           "pls": "please",

&#x20;           "plz": "please",

&#x20;           "thx": "thanks",

&#x20;           "bcz": "because",

&#x20;           "bcoz": "because",

&#x20;           "btwn": "between",

&#x20;           "diff": "difference",

&#x20;           "hw": "how",

&#x20;           "wht": "what",

&#x20;           "abt": "about",

&#x20;           "govt": "government",

&#x20;           "bio": "biology",

&#x20;           "chem": "chemistry",

&#x20;           "phy": "physics",

&#x20;       }

&#x20;       

&#x20;       print("\[OK] Hinglish Normalizer initialized")

&#x20;   

&#x20;   def normalize\_spelling(self, text):

&#x20;       """Normalize common Hinglish spelling variations."""

&#x20;       print("\\n" + "="\*60)

&#x20;       print("SPELLING NORMALIZATION")

&#x20;       print("="\*60)

&#x20;       print(f"  Input: \\"{text}\\"")

&#x20;       

&#x20;       words = text.split()

&#x20;       normalized\_words = \[]

&#x20;       changes = \[]

&#x20;       

&#x20;       for word in words:

&#x20;           # Preserve punctuation

&#x20;           prefix\_punct = ""

&#x20;           suffix\_punct = ""

&#x20;           clean\_word = word

&#x20;           

&#x20;           # Strip leading/trailing punctuation

&#x20;           while clean\_word and not clean\_word\[0].isalnum():

&#x20;               prefix\_punct += clean\_word\[0]

&#x20;               clean\_word = clean\_word\[1:]

&#x20;           while clean\_word and not clean\_word\[-1].isalnum():

&#x20;               suffix\_punct = clean\_word\[-1] + suffix\_punct

&#x20;               clean\_word = clean\_word\[:-1]

&#x20;           

&#x20;           # Check if word needs normalization

&#x20;           lower\_word = clean\_word.lower()

&#x20;           if lower\_word in self.reverse\_map:

&#x20;               normalized = self.reverse\_map\[lower\_word]

&#x20;               changes.append(f"'{clean\_word}' → '{normalized}'")

&#x20;               normalized\_words.append(prefix\_punct + normalized + suffix\_punct)

&#x20;           elif lower\_word in self.abbreviations:

&#x20;               normalized = self.abbreviations\[lower\_word]

&#x20;               changes.append(f"'{clean\_word}' → '{normalized}'")

&#x20;               normalized\_words.append(prefix\_punct + normalized + suffix\_punct)

&#x20;           else:

&#x20;               normalized\_words.append(word)

&#x20;       

&#x20;       result = " ".join(normalized\_words)

&#x20;       

&#x20;       if changes:

&#x20;           print(f"  Changes made:")

&#x20;           for change in changes:

&#x20;               print(f"    {change}")

&#x20;       else:

&#x20;           print(f"  No spelling changes needed")

&#x20;       

&#x20;       print(f"  Output: \\"{result}\\"")

&#x20;       

&#x20;       return result, changes

&#x20;   

&#x20;   def normalize\_sentence(self, text):

&#x20;       """

&#x20;       Full sentence normalization:

&#x20;       1. Case normalization (keep proper nouns)

&#x20;       2. Whitespace normalization

&#x20;       3. Punctuation normalization

&#x20;       4. Spelling normalization

&#x20;       """

&#x20;       print("\\n" + "="\*60)

&#x20;       print("SENTENCE NORMALIZATION (Complete)")

&#x20;       print("="\*60)

&#x20;       print(f"  Input: \\"{text}\\"")

&#x20;       

&#x20;       # Step 1: Whitespace normalization

&#x20;       text = re.sub(r'\\s+', ' ', text).strip()

&#x20;       

&#x20;       # Step 2: Punctuation normalization

&#x20;       text = re.sub(r'\\.{2,}', '.', text)     # Multiple dots → single

&#x20;       text = re.sub(r'\\!{2,}', '!', text)     # Multiple ! → single

&#x20;       text = re.sub(r'\\?{2,}', '?', text)     # Multiple ? → single

&#x20;       text = re.sub(r'\\s+(\[.,!?;:])', r'\\1', text)  # Remove space before punctuation

&#x20;       

&#x20;       # Step 3: Spelling normalization

&#x20;       text, \_ = self.normalize\_spelling(text)

&#x20;       

&#x20;       print(f"  Final output: \\"{text}\\"")

&#x20;       

&#x20;       return text





class WordLevelLanguageIdentifier:

&#x20;   """

&#x20;   Identify the language of each word in a Hinglish sentence.

&#x20;   Labels: HI (Hindi), EN (English), NE (Named Entity), UNIV (Universal), MIX (Mixed)

&#x20;   

&#x20;   This is a RULE-BASED approach for the initial prototype.

&#x20;   Later, this will be replaced with a fine-tuned MuRIL model.

&#x20;   """

&#x20;   

&#x20;   def \_\_init\_\_(self):

&#x20;       # Common Hindi words in Roman script

&#x20;       self.hindi\_words = {

&#x20;           # Pronouns

&#x20;           "mein", "main", "hum", "tum", "woh", "yeh", "ye", "wo",

&#x20;           "uska", "uski", "iska", "iski", "mera", "meri", "tera", "teri",

&#x20;           "unka", "unki", "hamara", "hamari", "tumhara", "tumhari",

&#x20;           

&#x20;           # Verbs / Auxiliaries

&#x20;           "hai", "hain", "tha", "thi", "the", "hoga", "hogi",

&#x20;           "hota", "hoti", "hote", "karta", "karti", "karte",

&#x20;           "kiya", "kiye", "ki", "karo", "karna", "karke",

&#x20;           "hona", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",

&#x20;           "aata", "aati", "aate", "jaata", "jaati", "jaate",

&#x20;           "deta", "deti", "dete", "leta", "leti", "lete",

&#x20;           "bana", "bani", "bane", "banta", "banti", "bante",

&#x20;           "samjhao", "samjho", "batao", "bolo", "dekho", "suno",

&#x20;           "likhna", "padhna", "seekhna",

&#x20;           "kehte", "kehta", "kehti", "kaha",

&#x20;           

&#x20;           # Postpositions

&#x20;           "ka", "ki", "ke", "ko", "se", "mein", "par", "tak",

&#x20;           "pe", "ne", "me",

&#x20;           

&#x20;           # Conjunctions / Connectors

&#x20;           "aur", "ya", "lekin", "kyunki", "isliye", "jabki",

&#x20;           "phir", "toh", "bhi", "hi", "sirf", "bas",

&#x20;           

&#x20;           # Question words

&#x20;           "kya", "kaise", "kyun", "kahan", "kab", "kaun", "kitna",

&#x20;           "kitni", "kitne", "konsa", "konsi",

&#x20;           

&#x20;           # Adjectives / Adverbs

&#x20;           "bahut", "thoda", "zyada", "kam", "accha", "bura",

&#x20;           "bada", "bade", "badi", "chhota", "chhoti", "chhote",

&#x20;           "naya", "nayi", "naye", "purana", "purani", "purane",

&#x20;           "pehle", "baad", "andar", "bahar", "upar", "neeche",

&#x20;           "yahan", "wahan", "abhi", "tab", "jab",

&#x20;           

&#x20;           # Negation

&#x20;           "nahi", "nhi", "na", "mat",

&#x20;           

&#x20;           # Others

&#x20;           "ek", "do", "teen", "chaar", "paanch",

&#x20;           "sabhi", "sab", "kuch", "koi",

&#x20;           "wala", "wale", "wali",

&#x20;           "jaise", "tarah", "matlab", "yaani",

&#x20;           "saath", "taraf", "beech",

&#x20;           "kaam", "cheez", "jagah", "tarika",

&#x20;           "dijiye", "dena", "lena",

&#x20;           "hokar", "jinmein", "jismein", "inmein",

&#x20;       }

&#x20;       

&#x20;       # Common English words that appear in Hinglish 

&#x20;       # (beyond what NLTK's word list covers)

&#x20;       self.common\_english\_indicators = {

&#x20;           "the", "is", "are", "was", "were", "been", "being",

&#x20;           "have", "has", "had", "do", "does", "did",

&#x20;           "will", "would", "could", "should", "may", "might",

&#x20;           "shall", "can", "must",

&#x20;           "and", "or", "but", "if", "then", "because",

&#x20;           "which", "that", "this", "these", "those",

&#x20;           "what", "where", "when", "how", "why", "who",

&#x20;           "not", "no", "yes",

&#x20;           "with", "without", "between", "through",

&#x20;           "from", "into", "onto", "upon",

&#x20;       }

&#x20;       

&#x20;       # Named entities / Universal words (language-independent)

&#x20;       self.universal\_indicators = {

&#x20;           "sir", "madam", "ok", "okay", "hello", "hi", "bye",

&#x20;           "please", "thank", "thanks", "sorry",

&#x20;       }

&#x20;       

&#x20;       # Science-specific English terms (should ALWAYS be tagged EN)

&#x20;       self.science\_terms = {

&#x20;           "cell", "cells", "membrane", "nucleus", "chromosome", "chromosomes",

&#x20;           "gene", "genes", "dna", "rna", "protein", "proteins",

&#x20;           "mitochondria", "ribosome", "ribosomes", "lysosome", "lysosomes",

&#x20;           "endoplasmic", "reticulum", "golgi", "apparatus", "vacuole", "vacuoles",

&#x20;           "plastid", "plastids", "chloroplast", "chloroplasts", "chlorophyll",

&#x20;           "osmosis", "diffusion", "permeable", "selectively",

&#x20;           "prokaryotic", "eukaryotic", "organism", "organisms",

&#x20;           "tissue", "tissues", "organ", "organs",

&#x20;           "photosynthesis", "respiration", "atp",

&#x20;           "microscope", "cork", "slice",

&#x20;           "structural", "functional", "fundamental",

&#x20;           "nuclear", "cytoplasm", "protoplasm",

&#x20;           "turgidity", "rigidity",

&#x20;           "inheritance", "offspring",

&#x20;           "concentration", "solution",

&#x20;           "energy", "power", "force",

&#x20;           "discovery", "discover", "observe",

&#x20;           "package", "dispatch", "transport",

&#x20;           "suicide", "bags",  # suicide bags = lysosomes

&#x20;           "pigment", "pigments",

&#x20;           "wall",  # cell wall

&#x20;           "plant", "animal",

&#x20;           "living", "life", "unit",

&#x20;           "function", "structure", "process",

&#x20;           "movement", "transfer", "control",

&#x20;           "region", "form", "type",

&#x20;           "produce", "provide", "contain", "contains",

&#x20;           "present", "absent",

&#x20;           "large", "small",

&#x20;           "well-defined", "well-organised",

&#x20;           "high", "low",

&#x20;       }

&#x20;       

&#x20;       print("\[OK] Word-Level Language Identifier initialized")

&#x20;   

&#x20;   def identify\_word\_language(self, word):

&#x20;       """Identify the language of a single word."""

&#x20;       clean\_word = re.sub(r'\[^\\w]', '', word).lower()

&#x20;       

&#x20;       if not clean\_word:

&#x20;           return "PUNCT"

&#x20;       

&#x20;       # Check if it contains Devanagari characters

&#x20;       has\_devanagari = any('\\u0900' <= c <= '\\u097F' for c in clean\_word)

&#x20;       has\_latin = any(c.isascii() and c.isalpha() for c in clean\_word)

&#x20;       

&#x20;       if has\_devanagari and has\_latin:

&#x20;           return "MIX"

&#x20;       elif has\_devanagari:

&#x20;           return "HI"

&#x20;       

&#x20;       # For Roman script, use lexicon lookup

&#x20;       if clean\_word.isdigit():

&#x20;           return "UNIV"

&#x20;       

&#x20;       if clean\_word in self.universal\_indicators:

&#x20;           return "UNIV"

&#x20;       

&#x20;       if clean\_word in self.science\_terms:

&#x20;           return "EN"

&#x20;       

&#x20;       if clean\_word in self.hindi\_words:

&#x20;           return "HI"

&#x20;       

&#x20;       if clean\_word in self.common\_english\_indicators:

&#x20;           return "EN"

&#x20;       

&#x20;       # Default heuristic: if it looks like English (common patterns)

&#x20;       # Check if the word is likely English by checking character patterns

&#x20;       if len(clean\_word) > 2:

&#x20;           # Words ending in common English suffixes

&#x20;           english\_suffixes = \['tion', 'sion', 'ness', 'ment', 'able', 'ible',

&#x20;                             'ous', 'ive', 'ful', 'less', 'ing', 'ed', 'er',

&#x20;                             'est', 'ly', 'al', 'ity', 'ence', 'ance']

&#x20;           for suffix in english\_suffixes:

&#x20;               if clean\_word.endswith(suffix):

&#x20;                   return "EN"

&#x20;       

&#x20;       # If still uncertain, default to English 

&#x20;       # (since NCERT content is English-heavy)

&#x20;       return "EN"

&#x20;   

&#x20;   def identify\_sentence(self, sentence):

&#x20;       """

&#x20;       Perform word-level language identification on a full sentence.

&#x20;       """

&#x20;       print("\\n" + "="\*60)

&#x20;       print("WORD-LEVEL LANGUAGE IDENTIFICATION")

&#x20;       print("="\*60)

&#x20;       print(f"  Input: \\"{sentence}\\"")

&#x20;       

&#x20;       words = sentence.split()

&#x20;       results = \[]

&#x20;       

&#x20;       print(f"\\n  {'Word':<25} {'Language':<10}")

&#x20;       print(f"  {'-'\*25} {'-'\*10}")

&#x20;       

&#x20;       for word in words:

&#x20;           lang = self.identify\_word\_language(word)

&#x20;           results.append({"word": word, "language": lang})

&#x20;           

&#x20;           # Color-code in terminal (if supported)

&#x20;           if lang == "HI":

&#x20;               marker = "🟠"

&#x20;           elif lang == "EN":

&#x20;               marker = "🔵"

&#x20;           elif lang == "NE":

&#x20;               marker = "🟢"

&#x20;           elif lang == "UNIV":

&#x20;               marker = "⚪"

&#x20;           elif lang == "MIX":

&#x20;               marker = "🟡"

&#x20;           else:

&#x20;               marker = "⚫"

&#x20;           

&#x20;           print(f"  {word:<25} {lang:<10} {marker}")

&#x20;       

&#x20;       # Calculate Code-Mixing Index (CMI)

&#x20;       lang\_counts = Counter(r\["language"] for r in results)

&#x20;       hi\_count = lang\_counts.get("HI", 0)

&#x20;       en\_count = lang\_counts.get("EN", 0)

&#x20;       total\_lang = hi\_count + en\_count

&#x20;       

&#x20;       if total\_lang > 0:

&#x20;           cmi = (1 - max(hi\_count, en\_count) / total\_lang) \* 100

&#x20;       else:

&#x20;           cmi = 0

&#x20;       

&#x20;       print(f"\\n  Language Distribution:")

&#x20;       print(f"    Hindi (HI):    {hi\_count} words")

&#x20;       print(f"    English (EN):  {en\_count} words")

&#x20;       print(f"    Universal:     {lang\_counts.get('UNIV', 0)} words")

&#x20;       print(f"    Named Entity:  {lang\_counts.get('NE', 0)} words")

&#x20;       print(f"    Mixed script:  {lang\_counts.get('MIX', 0)} words")

&#x20;       print(f"\\n  Code-Mixing Index (CMI): {cmi:.1f}%")

&#x20;       

&#x20;       if cmi == 0:

&#x20;           print(f"  → Monolingual text")

&#x20;       elif cmi < 30:

&#x20;           print(f"  → Low code-mixing")

&#x20;       elif cmi < 50:

&#x20;           print(f"  → Moderate code-mixing (typical Hinglish)")

&#x20;       else:

&#x20;           print(f"  → High code-mixing")

&#x20;       

&#x20;       return {

&#x20;           "words": results,

&#x20;           "cmi": round(cmi, 1),

&#x20;           "language\_distribution": dict(lang\_counts)

&#x20;       }





\# ============================================

\# USAGE

\# ============================================

if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   # === Script Detection ===

&#x20;   detector = ScriptDetector()

&#x20;   

&#x20;   # Test with different script types

&#x20;   test\_inputs = \[

&#x20;       "Sabhi living organisms cells se bane hote hain.",           # Roman

&#x20;       "सभी living organisms cells से बने होते हैं।",             # Mixed (Dev + Roman)

&#x20;       "Cell membrane ek selectively permeable membrane hoti hai.",  # Roman Hinglish

&#x20;       "All living organisms are made up of cells.",                 # Pure English

&#x20;   ]

&#x20;   

&#x20;   print("\\n" + "#"\*70)

&#x20;   print("# TESTING SCRIPT DETECTION")

&#x20;   print("#"\*70)

&#x20;   

&#x20;   for text in test\_inputs:

&#x20;       detector.detect\_text\_script(text)

&#x20;   

&#x20;   # === Normalization ===

&#x20;   normalizer = HinglishNormalizer()

&#x20;   

&#x20;   test\_normalize = \[

&#x20;       "Sir, osmosis kia hota h? plz smjhao",

&#x20;       "cell membrane ki function btao",

&#x20;       "mitochondria ko powerhouse kyu kehte he?",

&#x20;   ]

&#x20;   

&#x20;   print("\\n" + "#"\*70)

&#x20;   print("# TESTING NORMALIZATION")

&#x20;   print("#"\*70)

&#x20;   

&#x20;   for text in test\_normalize:

&#x20;       normalizer.normalize\_sentence(text)

&#x20;   

&#x20;   # === Language Identification ===

&#x20;   lid = WordLevelLanguageIdentifier()

&#x20;   

&#x20;   test\_lid = \[

&#x20;       "Sabhi living organisms cells se bane hote hain.",

&#x20;       "Sir, endoplasmic reticulum ka cell mein kya function hota hai?",

&#x20;       "Mitochondria ko cell ka powerhouse kaha jaata hai kyunki yeh ATP ke form mein energy produce karte hain.",

&#x20;       "All living organisms are made up of cells.",  # Pure English for comparison

&#x20;   ]

&#x20;   

&#x20;   print("\\n" + "#"\*70)

&#x20;   print("# TESTING LANGUAGE IDENTIFICATION")

&#x20;   print("#"\*70)

&#x20;   

&#x20;   for text in test\_lid:

&#x20;       lid.identify\_sentence(text)

```



\---



\## STEP 5: UNIFIED PIPELINE (Both — Together)



\### File: `src/pipeline.py`



```python

"""

EduHinglish - Unified Preprocessing Pipeline

Authors: Ashvatth \& Jatin

Purpose: Complete pipeline that processes BOTH English and Hinglish text

&#x20;        through all preprocessing stages and shows comparison

"""



import json

import os

import sys

from datetime import datetime



\# Import our modules

from preprocessing import EnglishPreprocessor

from script\_detector import ScriptDetector, HinglishNormalizer, WordLevelLanguageIdentifier





class EduHinglishPipeline:

&#x20;   """

&#x20;   Complete preprocessing pipeline for EduHinglish.

&#x20;   

&#x20;   For English text:

&#x20;     Text → Sentence Segmentation → Tokenization → Stop Words → 

&#x20;     Stemming → Lemmatization → POS Tagging

&#x20;   

&#x20;   For Hinglish text:

&#x20;     Text → Script Detection → Normalization → Language ID → 

&#x20;     Tokenization → Stop Words (language-aware) → Stemming → 

&#x20;     Lemmatization → POS Tagging

&#x20;   """

&#x20;   

&#x20;   def \_\_init\_\_(self):

&#x20;       print("="\*70)

&#x20;       print("INITIALIZING EduHinglish Preprocessing Pipeline")

&#x20;       print("="\*70)

&#x20;       

&#x20;       self.preprocessor = EnglishPreprocessor()

&#x20;       self.script\_detector = ScriptDetector()

&#x20;       self.normalizer = HinglishNormalizer()

&#x20;       self.lid = WordLevelLanguageIdentifier()

&#x20;       

&#x20;       print("\\n\[OK] All modules loaded successfully!")

&#x20;       print("="\*70)

&#x20;   

&#x20;   def process\_english(self, sentence):

&#x20;       """Process a pure English NCERT sentence."""

&#x20;       print("\\n" + "█"\*70)

&#x20;       print("█ MODE: ENGLISH PROCESSING")

&#x20;       print(f"█ Input: \\"{sentence\[:60]}...\\"")

&#x20;       print("█"\*70)

&#x20;       

&#x20;       result = {

&#x20;           "mode": "english",

&#x20;           "input": sentence,

&#x20;           "steps": {}

&#x20;       }

&#x20;       

&#x20;       # Run English preprocessing pipeline

&#x20;       pipeline\_result = self.preprocessor.process\_sentence(sentence)

&#x20;       result\["steps"] = pipeline\_result\["steps"]

&#x20;       

&#x20;       return result

&#x20;   

&#x20;   def process\_hinglish(self, sentence):

&#x20;       """Process a Hinglish (code-mixed) sentence."""

&#x20;       print("\\n" + "█"\*70)

&#x20;       print("█ MODE: HINGLISH PROCESSING")

&#x20;       print(f"█ Input: \\"{sentence\[:60]}...\\"")

&#x20;       print("█"\*70)

&#x20;       

&#x20;       result = {

&#x20;           "mode": "hinglish",

&#x20;           "input": sentence,

&#x20;           "steps": {}

&#x20;       }

&#x20;       

&#x20;       # Step 1: Script Detection

&#x20;       print("\\n--- HINGLISH STEP 1/6: Script Detection ---")

&#x20;       script\_result = self.script\_detector.detect\_text\_script(sentence)

&#x20;       result\["steps"]\["script\_detection"] = script\_result

&#x20;       

&#x20;       # Step 2: Normalization

&#x20;       print("\\n--- HINGLISH STEP 2/6: Normalization ---")

&#x20;       normalized = self.normalizer.normalize\_sentence(sentence)

&#x20;       result\["steps"]\["normalization"] = {

&#x20;           "input": sentence,

&#x20;           "output": normalized

&#x20;       }

&#x20;       

&#x20;       # Step 3: Language Identification

&#x20;       print("\\n--- HINGLISH STEP 3/6: Word-Level Language ID ---")

&#x20;       lid\_result = self.lid.identify\_sentence(normalized)

&#x20;       result\["steps"]\["language\_identification"] = lid\_result

&#x20;       

&#x20;       # Step 4: Tokenization (using the normalized text)

&#x20;       print("\\n--- HINGLISH STEP 4/6: Tokenization ---")

&#x20;       tokens = self.preprocessor.tokenize(normalized)

&#x20;       result\["steps"]\["tokenization"] = tokens

&#x20;       

&#x20;       # Step 5: Language-Aware Stop Word Removal

&#x20;       print("\\n--- HINGLISH STEP 5/6: Stop Word Removal (Language-Aware) ---")

&#x20;       # For Hinglish, we need to be careful:

&#x20;       # - Remove English stop words from English segments

&#x20;       # - Remove Hindi stop words from Hindi segments

&#x20;       # - NEVER remove technical terms

&#x20;       

&#x20;       hindi\_stop\_words = {

&#x20;           "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein",

&#x20;           "ek", "yeh", "woh", "bhi", "hi", "toh",

&#x20;           "ne", "par", "pe",

&#x20;       }

&#x20;       

&#x20;       filtered = \[]

&#x20;       removed = \[]

&#x20;       

&#x20;       for token in tokens:

&#x20;           token\_lower = token.lower()

&#x20;           word\_lang = self.lid.identify\_word\_language(token)

&#x20;           

&#x20;           # If it's an English word, check English stop words

&#x20;           if word\_lang == "EN" and token\_lower in self.preprocessor.stop\_words:

&#x20;               if token\_lower not in self.preprocessor.protected\_words:

&#x20;                   removed.append((token, "EN\_STOP"))

&#x20;                   continue

&#x20;           

&#x20;           # If it's a Hindi word, check Hindi stop words

&#x20;           if word\_lang == "HI" and token\_lower in hindi\_stop\_words:

&#x20;               removed.append((token, "HI\_STOP"))

&#x20;               continue

&#x20;           

&#x20;           # If it's punctuation only

&#x20;           if not token.isalnum() and len(token) == 1:

&#x20;               removed.append((token, "PUNCT"))

&#x20;               continue

&#x20;           

&#x20;           filtered.append(token)

&#x20;       

&#x20;       print(f"  Removed: {removed}")

&#x20;       print(f"  Kept: {filtered}")

&#x20;       result\["steps"]\["stop\_word\_removal"] = {

&#x20;           "kept": filtered,

&#x20;           "removed": removed

&#x20;       }

&#x20;       

&#x20;       # Step 6: Stemming \& Lemmatization (only on English words)

&#x20;       print("\\n--- HINGLISH STEP 6/6: Stemming \& Lemmatization (English words only) ---")

&#x20;       stems\_and\_lemmas = \[]

&#x20;       

&#x20;       print(f"\\n  {'Token':<20} {'Language':<8} {'Stem':<20} {'Lemma':<20}")

&#x20;       print(f"  {'-'\*20} {'-'\*8} {'-'\*20} {'-'\*20}")

&#x20;       

&#x20;       for token in filtered:

&#x20;           word\_lang = self.lid.identify\_word\_language(token)

&#x20;           

&#x20;           if word\_lang == "EN":

&#x20;               stem = self.preprocessor.stemmer\_porter.stem(token)

&#x20;               lemma = self.preprocessor.lemmatizer.lemmatize(token.lower())

&#x20;           else:

&#x20;               # Don't stem/lemmatize Hindi words (would be meaningless)

&#x20;               stem = token

&#x20;               lemma = token

&#x20;           

&#x20;           stems\_and\_lemmas.append({

&#x20;               "token": token,

&#x20;               "language": word\_lang,

&#x20;               "stem": stem,

&#x20;               "lemma": lemma

&#x20;           })

&#x20;           

&#x20;           marker = "←" if stem != token.lower() or lemma != token.lower() else ""

&#x20;           print(f"  {token:<20} {word\_lang:<8} {stem:<20} {lemma:<20} {marker}")

&#x20;       

&#x20;       result\["steps"]\["stemming\_lemmatization"] = stems\_and\_lemmas

&#x20;       

&#x20;       return result

&#x20;   

&#x20;   def process\_auto(self, sentence):

&#x20;       """

&#x20;       Automatically detect if input is English or Hinglish,

&#x20;       then route to appropriate pipeline.

&#x20;       """

&#x20;       # Quick check: does it contain Hindi words?

&#x20;       script\_info = self.script\_detector.detect\_text\_script(sentence)

&#x20;       

&#x20;       # Also check for Roman-script Hindi words

&#x20;       words = sentence.lower().split()

&#x20;       hindi\_word\_count = sum(1 for w in words 

&#x20;                             if self.lid.identify\_word\_language(w) == "HI")

&#x20;       

&#x20;       if (script\_info\["overall\_script"] in \["DEVANAGARI", "MIXED"] or 

&#x20;           hindi\_word\_count > 0):

&#x20;           return self.process\_hinglish(sentence)

&#x20;       else:

&#x20;           return self.process\_english(sentence)

&#x20;   

&#x20;   def compare\_english\_hinglish(self, english\_sentence, hinglish\_sentence):

&#x20;       """

&#x20;       Process both versions and compare side by side.

&#x20;       This is what the mentor wants to see!

&#x20;       """

&#x20;       print("\\n" + "▓"\*70)

&#x20;       print("▓ COMPARISON: English vs Hinglish Processing")

&#x20;       print("▓"\*70)

&#x20;       

&#x20;       print(f"\\n  English: \\"{english\_sentence}\\"")

&#x20;       print(f"  Hinglish: \\"{hinglish\_sentence}\\"")

&#x20;       

&#x20;       # Process both

&#x20;       en\_result = self.process\_english(english\_sentence)

&#x20;       hi\_result = self.process\_hinglish(hinglish\_sentence)

&#x20;       

&#x20;       # Side-by-side comparison summary

&#x20;       print("\\n" + "="\*70)

&#x20;       print("COMPARISON SUMMARY")

&#x20;       print("="\*70)

&#x20;       

&#x20;       print(f"\\n{'Aspect':<30} {'English':<35} {'Hinglish':<35}")

&#x20;       print(f"{'-'\*30} {'-'\*35} {'-'\*35}")

&#x20;       

&#x20;       en\_tokens = en\_result\["steps"]\["tokenization"]

&#x20;       hi\_tokens = hi\_result\["steps"]\["tokenization"]

&#x20;       print(f"{'Total Tokens':<30} {len(en\_tokens):<35} {len(hi\_tokens):<35}")

&#x20;       

&#x20;       en\_kept = en\_result\["steps"]\["stop\_word\_removal"]\["kept"]

&#x20;       hi\_kept = hi\_result\["steps"]\["stop\_word\_removal"]\["kept"]

&#x20;       print(f"{'After Stop Word Removal':<30} {len(en\_kept):<35} {len(hi\_kept):<35}")

&#x20;       

&#x20;       # CMI for Hinglish

&#x20;       cmi = hi\_result\["steps"]\["language\_identification"]\["cmi"]

&#x20;       print(f"{'Code-Mixing Index':<30} {'N/A (monolingual)':<35} {str(cmi) + '%':<35}")

&#x20;       

&#x20;       # Script

&#x20;       script = hi\_result\["steps"]\["script\_detection"]\["overall\_script"]

&#x20;       print(f"{'Script':<30} {'ROMAN':<35} {script:<35}")

&#x20;       

&#x20;       print("="\*70)

&#x20;       

&#x20;       return {

&#x20;           "english": en\_result,

&#x20;           "hinglish": hi\_result

&#x20;       }

&#x20;   

&#x20;   def save\_results(self, results, output\_path):

&#x20;       """Save pipeline results."""

&#x20;       os.makedirs(os.path.dirname(output\_path), exist\_ok=True)

&#x20;       

&#x20;       # Make JSON-serializable

&#x20;       serializable = json.loads(

&#x20;           json.dumps(results, default=str, ensure\_ascii=False)

&#x20;       )

&#x20;       

&#x20;       with open(output\_path, 'w', encoding='utf-8') as f:

&#x20;           json.dump(serializable, f, indent=2, ensure\_ascii=False)

&#x20;       print(f"\\n\[SAVED] Results → {output\_path}")





\# ============================================

\# MAIN DEMO — Run this to show the full pipeline

\# ============================================

if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   # Initialize the pipeline

&#x20;   pipeline = EduHinglishPipeline()

&#x20;   

&#x20;   # ============================================

&#x20;   # DEMO 1: Process 1-2 sentences (mentor's instruction)

&#x20;   # ============================================

&#x20;   

&#x20;   print("\\n" + "▓"\*70)

&#x20;   print("▓ DEMO 1: Processing individual sentences")

&#x20;   print("▓"\*70)

&#x20;   

&#x20;   # English NCERT sentence

&#x20;   eng\_sent = "The cell membrane is a selectively permeable membrane that controls the movement of substances into and out of the cell."

&#x20;   

&#x20;   # Same concept in Hinglish

&#x20;   hin\_sent = "Cell membrane ek selectively permeable membrane hoti hai jo substances ke movement ko cell ke andar aur bahar control karti hai."

&#x20;   

&#x20;   # Compare both

&#x20;   comparison = pipeline.compare\_english\_hinglish(eng\_sent, hin\_sent)

&#x20;   

&#x20;   # ============================================

&#x20;   # DEMO 2: Process a student query

&#x20;   # ============================================

&#x20;   

&#x20;   print("\\n" + "▓"\*70)

&#x20;   print("▓ DEMO 2: Processing a student's Hinglish query")

&#x20;   print("▓"\*70)

&#x20;   

&#x20;   student\_query = "Sir, endoplasmic reticulum ka cell mein kya function hota hai?"

&#x20;   result = pipeline.process\_hinglish(student\_query)

&#x20;   

&#x20;   # ============================================

&#x20;   # DEMO 3: Process with spelling errors (tests normalization)

&#x20;   # ============================================

&#x20;   

&#x20;   print("\\n" + "▓"\*70)

&#x20;   print("▓ DEMO 3: Processing query with spelling variations")

&#x20;   print("▓"\*70)

&#x20;   

&#x20;   messy\_query = "Sir osmosis kia hota h? plz smjhao"

&#x20;   result = pipeline.process\_hinglish(messy\_query)

&#x20;   

&#x20;   # ============================================

&#x20;   # DEMO 4: Mixed script input

&#x20;   # ============================================

&#x20;   

&#x20;   print("\\n" + "▓"\*70)

&#x20;   print("▓ DEMO 4: Processing mixed-script input")

&#x20;   print("▓"\*70)

&#x20;   

&#x20;   mixed\_input = "Mitochondria को cell ka powerhouse kaha jaata hai"

&#x20;   result = pipeline.process\_hinglish(mixed\_input)

&#x20;   

&#x20;   # Save all results

&#x20;   all\_results = {

&#x20;       "timestamp": datetime.now().isoformat(),

&#x20;       "project": "EduHinglish",

&#x20;       "module": "Preprocessing Pipeline",

&#x20;       "demos": {

&#x20;           "demo1\_comparison": comparison,

&#x20;           "demo2\_student\_query": result,

&#x20;       }

&#x20;   }

&#x20;   

&#x20;   pipeline.save\_results(all\_results, "../outputs/pipeline\_results/full\_pipeline\_demo.json")

&#x20;   

&#x20;   print("\\n" + "="\*70)

&#x20;   print("ALL DEMOS COMPLETE!")

&#x20;   print("="\*70)

```



\---



\## STEP 6: HOW TO RUN EVERYTHING



\### Sequence of Commands



```bash

\# 1. Navigate to project directory

cd EduHinglish/



\# 2. Activate virtual environment

source eduhinglish\_env/bin/activate



\# 3. First, Ashvatth extracts the PDF (needs the actual PDF file)

cd src/

python pdf\_extractor.py



\# 4. Jatin creates the Hinglish dataset

python hinglish\_dataset\_creator.py



\# 5. Ashvatth tests English preprocessing

python preprocessing.py



\# 6. Jatin tests script detection \& LID

python script\_detector.py



\# 7. TOGETHER — Run the full pipeline

python pipeline.py

```



\---



\## WHAT TO SHOW IN YOUR PRESENTATION



Based on your mentor's presentation topics, here's what maps where:



```

PRESENTATION STRUCTURE

│

├── 1. INTRO (2 slides)

│   └── What is EduHinglish, the language gap problem

│

├── 2. LITERATURE SURVEY (2-3 slides)

│   └── Summary of 11 papers, key takeaways

│

├── 3. GAPS (2 slides)

│   └── The 7 gaps from your report

│

├── 4. OBJECTIVES (1 slide)

│   └── 5 modules you're building

│

└── 5. PREPROCESSING WORK (5-6 slides) ← THE NEW WORK

&#x20;   │

&#x20;   ├── Slide: Pipeline Architecture Diagram

&#x20;   │   └── Show the flow: Input → Script Detection → 

&#x20;   │       Normalization → LID → Tokenization → etc.

&#x20;   │

&#x20;   ├── Slide: Script Detection Demo

&#x20;   │   └── Show 3 inputs (Roman/Devanagari/Mixed) 

&#x20;   │       and detection results

&#x20;   │

&#x20;   ├── Slide: Normalization Demo

&#x20;   │   └── Show "kia" → "kya", "h" → "hai" corrections

&#x20;   │

&#x20;   ├── Slide: Language Identification Demo

&#x20;   │   └── Show a Hinglish sentence with color-coded 

&#x20;   │       HI/EN tags and CMI score

&#x20;   │

&#x20;   ├── Slide: Tokenization + Stemming + Lemmatization

&#x20;   │   └── Show the table of token → stem → lemma

&#x20;   │

&#x20;   └── Slide: English vs Hinglish Comparison

&#x20;       └── Show the same biology concept processed 

&#x20;           through both pipelines side by side

```



\---



\## IMMEDIATE NEXT STEPS (This Week)



```

DAY 1-2 (Ashvatth):

&#x20; ✅ Download NCERT Class 9 Science PDF

&#x20; ✅ Run pdf\_extractor.py → get clean text

&#x20; ✅ Run preprocessing.py on 2 sentences → verify outputs

&#x20; ✅ Fix any issues with the cleaned text



DAY 1-2 (Jatin):

&#x20; ✅ Create the 15 Hinglish sentences (extend if possible to 20-25)

&#x20; ✅ Run script\_detector.py → verify detection works

&#x20; ✅ Test normalizer on messy inputs

&#x20; ✅ Verify LID accuracy against your manual labels



DAY 3-4 (Together):

&#x20; ✅ Run pipeline.py → full demo

&#x20; ✅ Save outputs as JSON

&#x20; ✅ Create comparison tables for presentation

&#x20; ✅ Prepare presentation slides



DAY 5:

&#x20; ✅ Review with each other

&#x20; ✅ Practice demo

&#x20; ✅ Ready for mentor presentation

```



\*\*Start with `preprocessing.py` (Ashvatth) and `hinglish\_dataset\_creator.py` (Jatin) — those are independent and can be done in parallel. Then merge in `pipeline.py`.\*\*



Shall I help you with any specific part first — the actual PDF text for Chapter 5, more Hinglish sentences, the presentation slides, or debugging any code?

