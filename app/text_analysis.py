import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
from enum import Enum
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define an Enum for personal pronouns
class Pronoun(Enum):
    I = 'I'
    WE = 'we'
    YOU = 'you'
    HE = 'he'
    SHE = 'she'
    IT = 'it'
    THEY = 'they'
    US = 'us'
    THEM = 'them'


def get_unique_words(text):
    dict = {
      'unique', 'exclusive', 'only', 'revolutionary', 'innovative', 'leading',
      'first', 'best-in-class', 'superior', 'advanced', 'breakthrough',
      'ultimate', 'premium', 'finest', 'exceptional', 'unmatched',
      'unrivaled', 'outstanding', 'extraordinary', 'remarkable', 'unparalleled',
      'pioneering', 'cutting-edge', 'state-of-the-art', 'next-generation', 'compared', 'original', 'legacy'
  }
    unique_words = []
    doc = get_tokens(text)
    for token in doc:
      if token.text.lower() in dict:
        unique_words.append(token.text)
    return unique_words
         



def get_tokens(text: str):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(str(text))
    return doc



def get_words(text: str):
    words = []
    doc = get_tokens(text)
    for token in doc:
        if token.is_alpha:  # Only count actual words
            words.append(token.text)
    return words

def get_comparatives(text: str):
  comparatives = []
  for token in get_tokens(text):  
    if token.tag_ == 'JJR':
        comparatives.append(token.text)
  return comparatives

def get_superlatives(text: str):
  superlatives = []
  for token in get_tokens(text):  
    if token.tag_ == 'JJS':
        superlatives.append(token.text)
  return superlatives


def extract_adj_noun_pairs(text: str):
    doc = get_tokens(text)
    pairs = []
    for token in doc:
        if token.pos_ == "ADJ":
            if token.i < len(doc) - 1:  
                next_token = token.nbor(1)
                if next_token.pos_ == "NOUN":
                    pairs.append(f"{token.text} {next_token.text}")
    return pairs

def get_semantic_similarity(text, keyword):
    # Get embeddings
    text_embedding = model.encode([str(text)])[0]
    keyword_embedding = model.encode([str(keyword)])[0]
    
    # Calculate and return cosine similarity
    return np.dot(text_embedding, keyword_embedding) / (
        np.linalg.norm(text_embedding) * np.linalg.norm(keyword_embedding)
    )



# Function to get statistics of the most common personal pronoun
def get_dominant_pronoun_stats(transcript):
    # Extract pronouns from the transcript using regex
    found_pronouns = re.findall(r'\b(?:' + '|'.join([pronoun.value for pronoun in Pronoun]) + r')\b', transcript, flags=re.IGNORECASE)
    
    # Count the occurrences of each pronoun
    pronoun_counts = Counter(found_pronouns)
    
    # Find the most common pronoun and its count
    if pronoun_counts:
        most_common_pronoun, most_common_count = pronoun_counts.most_common(1)[0]
        total_pronouns = sum(pronoun_counts.values())
        relative_amount = (most_common_count / total_pronouns) * 100  # Calculate percentage
        return most_common_pronoun.lower(), most_common_count, relative_amount
    else:
        return None, 0, 0.0
