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
        return 'none', 0, 0.0
    
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

def process_pronoun_data(df, text_column):
    for idx, row in df.iterrows():
        text = row[text_column]
        most_common_pronoun, most_common_pronoun_count, most_common_pronoun_pct = get_dominant_pronoun_stats(text)
        df.at[idx, f'{text_column}_most_common_pronoun'] = most_common_pronoun
        df.at[idx, f'{text_column}_most_common_pronoun_count'] = most_common_pronoun_count
        df.at[idx, f'{text_column}_most_common_pronoun_pct'] = most_common_pronoun_pct
    return df
