import spacy

# Load spaCy model once at module level
nlp = spacy.load('en_core_web_sm')

# Define constant set of unique words
UNIQUE_WORDS = {
    'unique', 'exclusive', 'only', 'revolutionary', 'innovative', 'leading',
    'first', 'best-in-class', 'superior', 'advanced', 'breakthrough',
    'ultimate', 'premium', 'finest', 'exceptional', 'unmatched',
    'unrivaled', 'outstanding', 'extraordinary', 'remarkable', 'unparalleled',
    'pioneering', 'cutting-edge', 'state-of-the-art', 'next-generation', 'compared', 'original', 'legacy'
}

def get_tokens(text: str):
    return nlp(str(text))

def get_unique_words(text):
    doc = get_tokens(text)
    return [token.text for token in doc if token.text.lower() in UNIQUE_WORDS]

def get_words(text: str):
    doc = get_tokens(text)
    return [token.text for token in doc if token.is_alpha]

def get_comparatives(text: str):
    doc = get_tokens(text)
    return [token.text for token in doc if token.tag_ == 'JJR']

def get_superlatives(text: str):
    doc = get_tokens(text)
    return [token.text for token in doc if token.tag_ == 'JJS']