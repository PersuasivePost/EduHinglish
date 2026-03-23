"""
EduHinglish - NLTK Data Downloader
Run this once after setting up your environment.
Usage: python scripts/download_nltk_data.py
"""

import nltk
import sys

def download_nltk_data():
    packages = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
    ]

    print("Downloading NLTK data packages...")
    print("=" * 50)

    success = []
    failed = []

    for path, name in packages:
        try:
            nltk.download(name, quiet=True)
            print(f"  [OK] {name}")
            success.append(name)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(name)

    print("=" * 50)
    print(f"Downloaded: {len(success)}/{len(packages)}")

    if failed:
        print(f"Failed: {failed}")
        print("Try manually: python -c \"import nltk; nltk.download('<name>')\"")
        sys.exit(1)
    else:
        print("All NLTK data ready!")

if __name__ == "__main__":
    download_nltk_data()