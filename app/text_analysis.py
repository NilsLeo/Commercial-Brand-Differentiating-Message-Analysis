import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import nltk
from collections import Counter
from enum import Enum
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import ast
import pandas as pd
from collections import defaultdict
model = SentenceTransformer('all-MiniLM-L6-v2')
import logging
logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
nltk.download('all')
nlp = spacy.load("en_core_web_sm")
# Define an Enum for personal pronouns
class PersonalPronoun(Enum):
    I = 'I'
    WE = 'we'
    YOU = 'you'
    HE = 'he'
    SHE = 'she'
    IT = 'it'
    THEY = 'they'
    US = 'us'
    THEM = 'them'

# Define an Enum for possessive pronouns
class PossessivePronoun(Enum):
    MY = 'my'
    OUR = 'our'
    OURS = 'ours'
    YOUR = 'your'
    YOURS = 'yours'
    HIS = 'his'
    HER = 'her'
    ITS = 'its'
    THEIR = 'their'
    THEIRS = 'theirs'

    
def get_unique_words(text):
    dict = {
      ‘original’, 
      ‘exceptional’, 
      ‘singular’, 
      ‘rare’, 
      ‘unrivaled’, 
      ‘unmatched’, 
      ‘extraordinary’, 
      ‘iconic’, 
      ‘distinct’, 
      ‘uncommon’, 
      ‘remarkable’, 
      ‘irreplaceable’, 
      ‘incomparable’, 
      ‘unparalleled’, 
      ‘one-of-a-kind’, 
      ‘unrepeatable’, 
      ‘innovative’, 
      ‘bold’, 
      ‘noteworthy’, 
      ‘distinctive’, 
      ‘outstanding’, 
      ‘spectacular’, 
      ‘unprecedented’, 
      ‘inimitable’, 
      ‘visionary’, 
      ‘striking’, 
      ‘groundbreaking’, 
      ‘great’, 
      ‘fantastic’, 
      ‘unstoppable’, 
      ‘perfect’, 
      ‘premium’, 
      ‘revolutionary’, 
      ‘legendary’, 
      ‘superior’, 
      ‘world-class’, 
      ‘ultimate’, 
      ‘favorite’, 
      ‘cutting-edge’, 
      ‘incredible’, 
      ‘matchless’, 
      ‘exclusive’, 
      ‘one-and-only’, 
      ‘rarefied’, 
      ‘peerless’, 
      ‘unique’, 
      ‘exclusive’, 
      ‘only’, 
      ‘revolutionary’, 
      ‘innovative’, 
      ‘leading’, 
      ‘first’, 
      ‘best-in-class’, 
      ‘superior’, 
      ‘advanced’, 
      ‘breakthrough’, 
      ‘ultimate’, 
      ‘premium’, 
      ‘finest’, 
      ‘exceptional’, 
      ‘unmatched’, 
      ‘unrivaled’, 
      ‘outstanding’, 
      ‘extraordinary’, 
      ‘remarkable’, 
      ‘unparalleled’, 
      ‘pioneering’, 
      ‘cutting-edge’, 
      ‘state-of-the-art’, 
      ‘next-generation’, 
      ‘compared’, 
      ‘original’, 
      ‘legacy’
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
    # Convert text to string to ensure compatibility with strip and other string methods
    text = str(text)
    # Check if the text is empty and return 0 if true
    if text.strip() == "":
        return 0

    # Get embeddings
    text_embedding = model.encode([text])[0]
    keyword_embedding = model.encode([str(keyword)])[0]
    
    # Calculate and return cosine similarity
    return np.dot(text_embedding, keyword_embedding) / (
        np.linalg.norm(text_embedding) * np.linalg.norm(keyword_embedding)
    )
def extract_keywords_list(df_column):
    keywords_list = []
    # Iterate over each row and collect keywords into a list
    for keywords in df_column:
        # Split the keywords string into a list of keywords and extend the main list
        individual_keywords = keywords.strip("[]").replace("'", "").split(', ')
        keywords_list.extend(individual_keywords)
    return keywords_list

def calculate_semantic_similarities(df, text_column, keyword_column):
    for idx, row in df.iterrows():
        # Skip rows where keyword column is null, NaN, or empty
        if pd.isna(row[keyword_column]) or row[keyword_column].strip() == "":
            continue

        text = row[text_column]
        logging.info(f"Keyword column: {row[keyword_column]}")
        keywords = row[keyword_column].strip("[]").replace("'", "").split(', ')
        keyword_similarities = {}
        
        logging.info(f"Calculating semantic similarities for {row['commercial_number']} in {text_column} with keywords: {keywords}")
        
        for keyword in keywords:
            logging.info(f"Calculating similarity for keyword: {keyword}")
            similarity = round(float(get_semantic_similarity(text, keyword)), 3)
            keyword_similarities[keyword] = similarity
        
        sorted_keywords = sorted(keyword_similarities.items(), key=lambda x: x[1], reverse=True)
        top_3_keywords = sorted_keywords[:3]
        top_3_average = round(float(np.mean([sim for _, sim in top_3_keywords])), 3) if top_3_keywords else 0
        
        logging.info(f"Top 3 keywords for {row['commercial_number']} in {text_column}:")
        for keyword, similarity in top_3_keywords:
            logging.info(f"- {keyword}: {similarity}")
        logging.info(f"Top 3 average similarity for {text_column}: {top_3_average}")
        
        df.at[idx, f'{text_column}_{keyword_column}_similarity'] = top_3_average
        df.at[idx, f'{text_column}_{keyword_column}_top_keywords'] = ', '.join([str(keyword) for keyword, _ in top_3_keywords])

    return df

# Function to get statistics of the most common personal pronoun
def process_text_data(df, text_column):
    df[f'{text_column}_word_count'] = 0
    df[f'{text_column}_superlative_count'] = 0
    df[f'{text_column}_superlative_pct'] = 0.0
    df[f'{text_column}_comparative_count'] = 0
    df[f'{text_column}_comparative_pct'] = 0.0
    df[f'{text_column}_uniqueness_count'] = 0
    df[f'{text_column}_uniqueness_pct'] = 0.0
    df[f'{text_column}_total_bdm_terms_count'] = 0
    df[f'{text_column}_total_bdm_terms_pct'] = 0.0
    for idx, row in df.iterrows():
        text = row[text_column]
        word_count = len(get_tokens(text))
        df.at[idx, f'{text_column}_word_count'] = word_count

        superlatives = get_superlatives(text)
        df.at[idx, f'{text_column}_superlatives'] = ', '.join(superlatives) if superlatives else ' '
        superlative_count = len(superlatives) if superlatives else 0
        df.at[idx, f'{text_column}_superlative_count'] = superlative_count

        comparatives = get_comparatives(text)
        df.at[idx, f'{text_column}_comparatives'] = ', '.join(comparatives) if comparatives else ' '
        comparative_count = len(comparatives) if comparatives else 0
        df.at[idx, f'{text_column}_comparative_count'] = comparative_count

        unique_words = get_unique_words(text)
        df.at[idx, f'{text_column}_unique_words'] = ', '.join(unique_words) if unique_words else ' '
        uniqueness_count = len(unique_words) if unique_words else 0
        df.at[idx, f'{text_column}_uniqueness_count'] = uniqueness_count

        if word_count > 0:
            df.at[idx, f'{text_column}_superlative_pct'] = superlative_count / word_count * 100
            df.at[idx, f'{text_column}_comparative_pct'] = comparative_count / word_count * 100
            df.at[idx, f'{text_column}_uniqueness_pct'] = uniqueness_count / word_count * 100

            total_bdm_terms = superlative_count + comparative_count + uniqueness_count
            df.at[idx, f'{text_column}_total_bdm_terms_count'] = total_bdm_terms
            df.at[idx, f'{text_column}_total_bdm_terms_pct'] = total_bdm_terms / word_count * 100

    return df




def find_first_modifier_with_degree(token, visited=None, first_modifier=None):
    if visited is None:
        visited = set()
    visited.add(token)

    # Blacklist für unerwünschte Modifier
    blacklist = {"old"}  # Füge hier weitere unerwünschte Wörter hinzu

    # Check if the next token is % directly after the number
    token_text = token.text
    has_percent = False
    try:
        next_token = token.nbor(1)  # Get the next token
        if next_token.text == "%":  # If the next token is %, combine it with the number
            token_text = "%"
            has_percent = True
    except IndexError:
        pass  # If there is no next token, skip this step

    # Check if the current token is an adjective or adverb with a defined Degree
    if token.pos_ in {"ADJ", "ADV"}:
        if token.text in blacklist:  # Skip blacklisted words
            return None
        degree = token.morph.get("Degree")
        if degree:  # Only return if Degree is defined (e.g., Cmp or Sup)
            modifier = f"{token.text}"
            if first_modifier is None or token.text != first_modifier:  # Ensure no repetition
                return modifier

    # Check children for modifiers
    for child in token.children:
        if child not in visited:
            modifier = find_first_modifier_with_degree(child, visited, first_modifier)
            if modifier:
                # Skip if the modifier is blacklisted
                if any(blacklisted_word in modifier for blacklisted_word in blacklist):
                    continue
                # Combine % with the modifier if found
                return f"{token_text} {modifier}" if has_percent else modifier

    # Check the head for modifiers
    if token.head not in visited and token.head != token:
        modifier = find_first_modifier_with_degree(token.head, visited, first_modifier)
        if modifier:
            # Skip if the modifier is blacklisted
            if any(blacklisted_word in modifier for blacklisted_word in blacklist):
                return None
            # Combine % with the modifier if found
            return f"{token_text} {modifier}" if has_percent else modifier

    # Return the token text with % if no modifier found
    return token_text if has_percent else None


def extract_features(text):
    doc = nlp(text)
    features = []
    for token in doc:
        if token.like_num:  # Check if the token is a number
            #print(f"Found number: {token.text}")
            # Find the first modifier related to the number
            modifier = find_first_modifier_with_degree(token)
            if modifier and modifier.lower() != token.text.lower():
                # Search for the modifier in the document
                if not any(tok.text == modifier and tok.like_num for tok in doc):
                    features.append((token.text, modifier))
    return features

def apply_on_transcript(text):
    # Check if the input is not a string or is empty
    if not isinstance(text, str) or not text:
        return []
    
    found_pairs = extract_features(text)
    return found_pairs

def contains_i(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'i' in re.findall(r'\bI\b', transcript, flags=re.IGNORECASE)

def contains_we(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'we' in re.findall(r'\bWE\b', transcript, flags=re.IGNORECASE)

def contains_you(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'you' in re.findall(r'\bYOU\b', transcript, flags=re.IGNORECASE)

def contains_he(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'he' in re.findall(r'\bHE\b', transcript, flags=re.IGNORECASE)

def contains_she(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'she' in re.findall(r'\bSHE\b', transcript, flags=re.IGNORECASE)

def contains_it(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'it' in re.findall(r'\bIT\b', transcript, flags=re.IGNORECASE)

def contains_they(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'they' in re.findall(r'\bTHEY\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'us' in the transcript
def contains_us(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'us' in re.findall(r'\bUS\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'them' in the transcript
def contains_them(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'them' in re.findall(r'\bTHEM\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'my' in the transcript
def contains_my(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'my' in re.findall(r'\bMY\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'our' in the transcript
def contains_our(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'our' in re.findall(r'\bOUR\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'ours' in the transcript
def contains_ours(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'ours' in re.findall(r'\bOURS\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'your' in the transcript
def contains_your(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'your' in re.findall(r'\bYOUR\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'yours' in the transcript
def contains_yours(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'yours' in re.findall(r'\bYOURS\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'his' in the transcript
def contains_his(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'his' in re.findall(r'\bHIS\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'her' in the transcript
def contains_her(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'her' in re.findall(r'\bHER\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'its' in the transcript
def contains_its(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'its' in re.findall(r'\bITS\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'their' in the transcript
def contains_their(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'their' in re.findall(r'\bTHEIR\b', transcript, flags=re.IGNORECASE)

# Function to check for the presence of 'theirs' in the transcript
def contains_theirs(transcript):
    if not isinstance(transcript, str) or not transcript:
        return False
    return 'theirs' in re.findall(r'\bTHEIRS\b', transcript, flags=re.IGNORECASE)

