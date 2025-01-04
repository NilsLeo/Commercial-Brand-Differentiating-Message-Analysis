#!/usr/bin/env python
# coding: utf-8

# # Detecting BDM In Superbowl Commercials

# In[94]:


import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
get_ipython().run_line_magic('pip', 'install -r requirements.txt')


# In[95]:


BDM_excel = pd.read_excel(f'{os.getenv("BDM_EXCEL_FILE")}')
final_excel = pd.read_excel(f'{os.getenv("FINAL_EXCEL_FILE")}')


# In[96]:


BDM_excel = pd.read_excel(f'{os.getenv("BDM_EXCEL_FILE")}')
final_excel = pd.read_excel(f'{os.getenv("FINAL_EXCEL_FILE")}')



# In[97]:


final_excel = final_excel.merge(
    BDM_excel[['AdNumber', 'BDM']], 
    on='AdNumber', 
    how='left',
    suffixes=('_old', '')
).drop('BDM_old', axis=1, errors='ignore')


# print number of rows where BDM is NaN, 0 and 1
print(f"Number of rows where BDM is NaN: {final_excel[final_excel['BDM'].isna()].shape[0]}")
print(f"Number of rows where BDM is 0: {final_excel[final_excel['BDM'] == 0].shape[0]}")
print(f"Number of rows where BDM is 1: {final_excel[final_excel['BDM'] == 1].shape[0]}")


# In[98]:


ad_df = final_excel.groupby(['cont_primary_product_type', 'BRAND', 'AdNumber', "BDM"]).size().reset_index(name='count')
ad_df.rename(columns={'cont_primary_product_type': 'product_category', 'BRAND': 'brand', 'AdNumber': 'commercial_number'}, inplace=True)
ad_df.drop(columns=['count'], inplace=True)
ad_df.head(10)


# ## Retrieving Transcript

# In[99]:


import glob
from pathlib import Path

# Get all txt files recursively from ADS_DIR
ads_dir = Path(os.getenv("ADS_DIR"))
transcript_files = glob.glob(str(ads_dir / "**/*.txt"), recursive=True)
# print transcript_files
print(transcript_files)
# Create a dictionary mapping commercial numbers to file paths
transcript_map = {Path(f).stem: f for f in transcript_files}

# Update transcripts in dataframe
ad_df['transcript'] = ''
for idx, row in ad_df.iterrows():
    commercial_num = row['commercial_number']
    if commercial_num in transcript_map:
        try:
            with open(transcript_map[commercial_num], 'r', encoding='utf-8') as f:
                ad_df.at[idx, 'transcript'] = f.read().strip()
        except FileNotFoundError:
            ad_df.at[idx, 'transcript'] = None
    else:
        ad_df.at[idx, 'transcript'] = None

ad_df[ad_df['transcript'].notna()]
ad_df.head(10)


# ## Adding OCR Text

# In[100]:


ocr_to_merge = pd.read_csv("./ocr_to_merge.csv")
ad_df = ad_df.merge(ocr_to_merge, left_on='commercial_number', right_on='ad', how='left')
ad_df.drop(columns=['ad', 'recognized_text'], inplace=True)
ad_df.rename(columns={'cleaned_text': 'ocr_text'}, inplace=True)

# merge ocr_text with transcript
# TODO: Rename transcript to transcript_plus_ocr
ad_df['transcript'] = ad_df['ocr_text'] + ' ' + ad_df['transcript']
ad_df.drop(columns=['ocr_text'], inplace=True)

ad_df.head()


# In[101]:


get_ipython().system('python -m spacy download en_core_web_sm')


# # Determining Frequency of Superlatives and Comparative Adjectives

# In[102]:


import spacy
import pandas as pd
from collections import Counter

# Load English language model
nlp = spacy.load('en_core_web_sm')

# Keywords that indicate uniqueness or superiority (expanded list)
uniqueness_terms = {
    'unique', 'exclusive', 'only', 'revolutionary', 'innovative', 'leading',
    'first', 'best-in-class', 'superior', 'advanced', 'breakthrough',
    'ultimate', 'premium', 'finest', 'exceptional', 'unmatched',
    'unrivaled', 'outstanding', 'extraordinary', 'remarkable', 'unparalleled',
    'pioneering', 'cutting-edge', 'state-of-the-art', 'next-generation', 'compared', 'original', 'legacy'
}

# Initialize lists to store percentages and words
analysis_results = []
identified_words = []

# Process each transcript
for transcript in ad_df['transcript']:
    # Initialize counters and word lists
    word_count = 0
    metrics = Counter()
    words = {
        'comparatives': [],
        'superlatives': [],
        'unique_words': [],
        'bdm_words': []
    }
    
    # Process the text with spaCy
    doc = nlp(str(transcript))
    
    # Analyze each token
    for token in doc:
        if token.is_alpha:  # Only count actual words
            word_count += 1
            
            if token.tag_ == 'JJR':
                metrics['comparative'] += 1
                words['comparatives'].append(token.text)
                words['bdm_words'].append(token.text)
            elif token.tag_ == 'JJS':
                metrics['superlative'] += 1
                words['superlatives'].append(token.text)
                words['bdm_words'].append(token.text)
            elif token.text.lower() in uniqueness_terms:
                metrics['uniqueness'] += 1
                words['unique_words'].append(token.text)
                words['bdm_words'].append(token.text)
    # Calculate percentages
    if word_count > 0:
        percentages = {
            'comparative_pct': (metrics['comparative'] / word_count) * 100,
            'superlative_pct': (metrics['superlative'] / word_count) * 100,
            'uniqueness_pct': (metrics['uniqueness'] / word_count) * 100,
            'total_bdm_terms_pct': sum(metrics.values()) / word_count * 100
        }
    else:
        percentages = {
            'comparative_pct': 0,
            'superlative_pct': 0,
            'uniqueness_pct': 0,
            'total_bdm_terms_pct': 0
        }
    
    analysis_results.append(percentages)
    identified_words.append(words)

# Add results to DataFrame
results_df = pd.DataFrame(analysis_results)
words_df = pd.DataFrame(identified_words)

# Combine all DataFrames
ad_df = pd.concat([
    ad_df, 
    results_df,
    words_df
], axis=1)

# Show summary statistics
print("\nPercentage Statistics:")
print(results_df.describe())

# Show correlation with BDM
print("\nCorrelation with BDM:")
for col in results_df.columns:
    correlation = ad_df[col].corr(ad_df['BDM'])
    print(f"{col}: {correlation:.3f}")
# sort by highest amount of superlatives, then by highest amount of comparatives
ad_df = ad_df.sort_values(by=['superlative_pct', 'comparative_pct', 'uniqueness_pct'], ascending=[False, False, False])
ad_df.head(10)


# In[103]:


# remove superlatives, comparatives and unique_words from ad_df
# TODO: Comment back in
# ad_df.drop(columns=['superlatives', 'comparatives', 'unique_words', 'bdm_words'], inplace=True)
ad_df.drop(columns=['comparative_pct', 'superlative_pct', 'uniqueness_pct'], inplace=True)
ad_df.head(10)


# # Product Category DF

# In[104]:


product_brands_df = pd.read_csv("product_categories.csv")
product_brands_df.head(40)
product_brands_df = product_brands_df.drop('product_cat_id', axis=1)
ad_df = ad_df.drop('product_category', axis=1)
display(product_brands_df)
display(ad_df)


# In[105]:


# Create a dictionary to map brands to their product categories and other attributes
brand_to_info = {}
for _, row in product_brands_df.iterrows():
    # Convert string representation of list to actual list
    brands = eval(row['product_cat_brands'])
    for brand in brands:
        # Remove spaces and convert to lowercase for more robust matching
        brand = brand.replace(' ', '').lower()
        # Store all columns for this brand
        brand_to_info[brand] = {col: row[col] for col in product_brands_df.columns}

# Function to find category info for a brand
def find_brand_info(brand):
    # Clean brand name for matching
    clean_brand = brand.replace(' ', '').lower()
    return brand_to_info.get(clean_brand)

# Add all product category columns to ad_df
for col in product_brands_df.columns:
    ad_df[col] = ad_df['brand'].apply(lambda x: find_brand_info(x)[col] if find_brand_info(x) else None)

# Print brands that couldn't be mapped
unmapped_brands = ad_df[ad_df['product_cat_name'].isna()]['brand'].unique()
if len(unmapped_brands) > 0:
    print("Brands without category mapping:")
    for brand in unmapped_brands:
        print(f"- {brand}")

# Print number of rows with missing category
print(f"Number of rows where product category is NaN: {ad_df[ad_df['product_cat_name'].isna()].shape[0]}")

# Drop rows with missing categories
ad_df = ad_df.dropna(subset=['product_cat_name'])

print(f"Final number of rows with missing categories: {ad_df[ad_df['product_cat_name'].isna()].shape[0]}")


# In[106]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict

nltk.download('all')


# In[107]:


from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
def get_semantic_similarity(text, keywords):
    # Convert string keywords list to actual list if needed
    if isinstance(keywords, str):
        keywords = eval(keywords)
    
    # Combine keywords into a single string
    keywords_text = ' '.join(keywords)
    
    # Get embeddings
    text_embedding = model.encode([str(text)])[0]
    keywords_embedding = model.encode([keywords_text])[0]
    
    # Calculate cosine similarity
    similarity = np.dot(text_embedding, keywords_embedding) / (
        np.linalg.norm(text_embedding) * np.linalg.norm(keywords_embedding)
    )
    
    # Find most similar phrases
    sentences = str(text).split('.')
    sentence_embeddings = model.encode(sentences)
    keyword_embedding = model.encode([keywords_text])[0]
    
    # Calculate similarities for each sentence
    sentence_similarities = np.dot(sentence_embeddings, keyword_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * 
        np.linalg.norm(keyword_embedding)
    )
    
    # Get top 3 most similar sentences
    top_indices = np.argsort(sentence_similarities)[-3:][::-1]
    similar_phrases = [sentences[i].strip() for i in top_indices if sentence_similarities[i] > 0.3]
    
    return similarity, similar_phrases

# Apply the analysis to the DataFrame
similarities = []
similar_phrases = []

for _, row in ad_df.iterrows():
    sim, phrases = get_semantic_similarity(row['transcript'], row['product_cat_keywords'])
    similarities.append(sim)
    similar_phrases.append(phrases)

# Add new columns to DataFrame
ad_df['keyword_similarity'] = similarities
ad_df['similar_phrases'] = similar_phrases


ad_df.head(10)


# In[108]:


# TODO: Implement proper handling of missing values

ad_df[ad_df.isnull().any(axis=1)].head()
ad_df[ad_df.isna().any(axis=1)].head()

ad_df = ad_df.dropna()
ad_df = ad_df[ad_df['transcript'] != '']
ad_df = ad_df[ad_df['transcript'] != '']

# print all from ad_df with empty values
print(ad_df[ad_df.isnull().any(axis=1)])
print(ad_df[ad_df.isna().any(axis=1)])


# In[109]:


ad_df.head(20)


# In[110]:


# show all rows ehere bdm is 1.0 and where industry is product_cat_id 4

nice_df = ad_df[ad_df['BDM'] == 1.0]


# Create a function to find shared words
def get_shared_words(row):
    # Convert keywords string to list if it's a string
    keywords = eval(row['product_cat_keywords']) if isinstance(row['product_cat_keywords'], str) else row['product_cat_keywords']
    
    # Convert all keywords to lowercase for better matching
    keywords = [word.lower() for word in keywords]
    
    # Split transcript into words and convert to lowercase
    transcript_words = set(word.lower() for word in str(row['transcript']).split())
    
    # Find intersection between keywords and transcript words
    shared = [word for word in keywords if word in transcript_words]
    
    return shared

# Add new column for shared words
ad_df['shared_keywords'] = ad_df.apply(get_shared_words, axis=1)

# Add column for count of shared words
ad_df['shared_keywords_count'] = ad_df['shared_keywords'].str.len()

# Update the display code
nice_df = ad_df[ad_df['BDM'] == 1.0]
display(nice_df)


# In[ ]:





# In[111]:


# Calculate the minimum number of samples in each group
# 
min_samples = min(len(ad_df[ad_df['BDM'] == 1]), len(ad_df[ad_df['BDM'] == 0]))

# Perform undersampling
ad_df_balanced = pd.concat([
    ad_df[ad_df['BDM'] == 1].sample(n=min_samples, random_state=42),
    ad_df[ad_df['BDM'] == 0].sample(n=min_samples, random_state=42)
]).reset_index(drop=True)

# Print the results
print(f"Total rows: {len(ad_df_balanced)}")
print(f"Rows with BDM = 1.0: {len(ad_df_balanced[ad_df_balanced['BDM'] == 1.0])}")
print(f"Rows with BDM = 0.0: {len(ad_df_balanced[ad_df_balanced['BDM'] == 0.0])}")



commercial_numbers = ad_df_balanced['commercial_number']


ad_df_balanced.head(20)


# ## Ansatz 1 (Machine learning)

# In[112]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
# Prepare features
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix, roc_curve, auc
# import gridsearchcv
from sklearn.model_selection import GridSearchCV




# In[113]:


# Prepare the data
data = ad_df_balanced[['keyword_similarity', 'total_bdm_terms_pct']]
target = ad_df_balanced['BDM']


# In[114]:


def get_base_models():
    """Return dictionary of base model configurations"""
    return {
        'Majority Classifier': DummyClassifier(),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(random_state=42),
    }


# In[115]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def get_param_distributions():
    """Return parameter distributions for RandomizedSearchCV"""
    return {
        'Logistic Regression': {
            'C': uniform(0.1, 10.0),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'Decision Tree': {
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        },
        'Random Forest': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20)
        },
        'Support Vector Machine': {
            'C': uniform(0.1, 20.0),
            'kernel': ['rbf', 'linear']
        },
        'Gradient Boosting': {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10)
        },
        'Dummy Classifier': {
            'strategy': ['stratified']
        }
    }

def tune_models(X, y, models, param_distributions, cv=5, n_iter=20):
    """Perform RandomizedSearchCV on specified models"""
    tuned_models = {}
    
    for name, model in models.items():
        if name in param_distributions:
            print(f"\nTuning {name}...")
            random_search = RandomizedSearchCV(
                model,
                param_distributions[name],
                n_iter=n_iter,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            random_search.fit(X, y)
            tuned_models[name] = random_search.best_estimator_
            print(f"Best parameters: {random_search.best_params_}")
            print(f"Best score: {random_search.best_score_:.3f}")
        else:
            tuned_models[name] = model
            
    return tuned_models

# Usage:
base_models = get_base_models()
param_distributions = get_param_distributions()
models = tune_models(data, target, base_models, param_distributions)


# In[116]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix,
    classification_report
)



# In[117]:


def evaluate_models(X, y, models, cv=5):
    """
    Evaluate models and add predictions to original dataframe
    """
    results = []
    predictions = pd.DataFrame()
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        model.fit(X, y)
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Create confusion matrix categories
        pred_categories = []
        for true, pred in zip(y, y_pred):
            if true == 0 and pred == 0:
                pred_categories.append('TN')
            elif true == 0 and pred == 1:
                pred_categories.append('FP')
            elif true == 1 and pred == 0:
                pred_categories.append('FN')
            else:  # true == 1 and pred == 1
                pred_categories.append('TP')
        
        predictions[f'{name}_result'] = pred_categories
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, 
            X, 
            y, 
            cv=cv, 
            scoring=['roc_auc', 'accuracy', 'precision', 'recall']
        )
        
        # Calculate metrics as before
        results.append({
            'Model': name,
            'ROC AUC (Mean)': np.mean(cv_results['test_roc_auc']),
            'ROC AUC (Std)': np.std(cv_results['test_roc_auc']),
            'Accuracy (Mean)': np.mean(cv_results['test_accuracy']),
            'Accuracy (Std)': np.std(cv_results['test_accuracy']),
            'Precision (Mean)': np.mean(cv_results['test_precision']),
            'Precision (Std)': np.std(cv_results['test_precision']),
            'Recall (Mean)': np.mean(cv_results['test_recall']),
            'Recall (Std)': np.std(cv_results['test_recall'])
        })
    
    results_df = pd.DataFrame(results).sort_values('ROC AUC (Mean)', ascending=False)
    return results_df, predictions


def plot_confusion_matrices(X, y, models):
    """
    Plot confusion matrices for each model
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - models: Dictionary of tuned models
    """
    n_models = len(models)
    fig, axes = plt.subplots(
        nrows=(n_models + 1) // 2, 
        ncols=2, 
        figsize=(16, 4 * ((n_models + 1) // 2))
    )
    axes = axes.ravel()  # Flatten axes array
    
    for i, (name, model) in enumerate(models.items()):
        # Predict using the model
        y_pred = model.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            ax=axes[i]
        )
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    # Remove extra subplots if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()



results_df, predictions = evaluate_models(data, target, models)
original_data = ad_df_balanced.copy()
original_data = pd.concat([original_data, commercial_numbers], axis=1)

#
# 
#  Add predictions to original data
original_data = pd.concat([original_data, predictions], axis=1)

# Display results
print("Cross-Validation Results:\n")
display(Markdown(results_df.to_markdown(index=False)))
plot_confusion_matrices(data, target, models)

# Display sample of data with predictions


# rename frame
predicted_data = original_data

# only include the top 3 models prediction results
predicted_data = predicted_data[['commercial_number', 'BDM', 'Logistic Regression_result', 'Random Forest_result', 'Support Vector Machine_result']]
# write the majority result of the colums logistic regression, random forest and support vector machine to a new column majority vote
predicted_data['majority_vote'] = predicted_data[['Logistic Regression_result', 'Random Forest_result', 'Support Vector Machine_result']].mode(axis=1)[0]
display(predicted_data.head(10))


linlu_table = predicted_data[['commercial_number', 'BDM', 'majority_vote']]
# only show those rows where the majority vote is FP or FN
linlu_table = linlu_table[(linlu_table['majority_vote'] == 'FP') | (linlu_table['majority_vote'] == 'FN')]
linlu_table['comment'] = ''
display(linlu_table.head(10))

# output to excel
linlu_table.to_excel('linlu_table.xlsx', index=False)


# In[118]:


from sklearn.tree import plot_tree

print("\nDecision Tree Analysis:")
simple_decision_tree = models['Decision Tree']
simple_decision_tree.fit(data, target)

# Create a new figure for the decision tree
plt.figure(figsize=(20, 10))
plot_tree(simple_decision_tree, 
          filled=True, 
          feature_names=data.columns, 
          class_names=['No BDM', 'BDM'])
plt.title("Simple Decision Tree Visualization")
plt.show()

# Print feature importance ranking
importances = simple_decision_tree.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance Ranking:")
for f in range(data.shape[1]):
    print(f"{f + 1}. {data.columns[indices[f]]}: {importances[indices[f]]:.3f}")


# ## Ansatz 2 - RNN + LSTM

# In[119]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow_hub as hub


# In[120]:


df = ad_df_balanced
df = df[['transcript', 'BDM']]
df = df.rename(columns={"transcript": "description", "BDM": "label"})
df['label'] = df['label'].astype(int)
df.head()

ad_df.info()


# In[121]:


train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])


# In[122]:


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('label')
    
    # Clean the descriptions - replace NaN with empty string and ensure all items are strings
    descriptions = df["description"].fillna("").astype(str).tolist()
    
    # Convert to tensors
    ds = tf.data.Dataset.from_tensor_slices((descriptions, labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


# In[123]:


train_data = df_to_dataset(train)
valid_data = df_to_dataset(val)
test_data = df_to_dataset(test)


# # Embedding + Model

# In[124]:


embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)


# In[125]:


hub_layer(list(train_data)[0][0])


# In[126]:


import tf_keras

model = tf_keras.Sequential([
    hub_layer,
    tf_keras.layers.Dense(16, activation='relu'),
    tf_keras.layers.Dropout(0.4),
    tf_keras.layers.Dense(16, activation='relu'),
    tf_keras.layers.Dropout(0.4),
    tf_keras.layers.Dense(1, activation='sigmoid')
])


# In[127]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


# In[128]:


model.evaluate(train_data)


# In[129]:


model.evaluate(valid_data)


# In[130]:


# history = model.fit(train_data, epochs=3, validation_data=valid_data)


# In[131]:


# 


# In[132]:


model.evaluate(test_data)


# In[133]:


# After the last cell (in[55]), add:

import seaborn as sns
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(model, test_data):
  # Get predictions for test data
  test_predictions = model.predict(test_data)
  test_predictions = (test_predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

  # Get true labels from test data
  test_labels = np.concatenate([y for x, y in test_data], axis=0)

  # Create confusion matrix
  cm = confusion_matrix(test_labels, test_predictions)

  # Plot confusion matrix
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.title('LSTM Model Confusion Matrix')
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.show()

  # Calculate and print metrics
  tn, fp, fn, tp = cm.ravel()
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1 = 2 * (precision * recall) / (precision + recall)

  print("\nLSTM Model Metrics:")
  print(f"Accuracy: {accuracy:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")
  print(f"F1 Score: {f1:.3f}")


# In[134]:


plot_confusion_matrix(model, test_data)


# # LSTM

# In[135]:


encoder = tf.keras.layers.TextVectorization(max_tokens=600)
encoder.adapt(train_data.map(lambda text, label: text))


# In[136]:


vocab = np.array(encoder.get_vocabulary())
vocab[:20]


# In[137]:


model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=32,
        mask_zero=True
    ),
    tf.keras.layers.LSTM(32, use_cudnn=False),  # Disable cuDNN
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


# In[138]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


# In[139]:


model.evaluate(train_data)
model.evaluate(valid_data)


# In[140]:


history = model.fit(train_data, epochs=5, validation_data=valid_data)


# In[141]:


model.evaluate(test_data)


# In[142]:


plot_confusion_matrix(model, test_data)

