import string
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # Download the Open Multilingual WordNet
from nltk.corpus import wordnet
from spellchecker import SpellChecker



def normalize_query(query):
    # Lowercase
    query = query.lower()
    # Remove punctuation
    query = query.translate(str.maketrans('', '', string.punctuation))
    return query

def expand_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = [spell.correction(word) for word in words]
    return ' '.join(corrected_words)

def preprocess_query(query):
    # normalize query
    normalized_query = normalize_query(query)
    # Correct spelling first
    corrected_query = correct_spelling(normalized_query)
    
    # Optionally, expand synonyms for key terms
    # This step is context-dependent and might not be applied universally
    words = corrected_query.split()
    expanded_query = " ".join([expand_synonyms(word)[0] if expand_synonyms(word) else word for word in words])
    
    return expanded_query