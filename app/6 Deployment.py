import streamlit as st
import pandas as pd  # Add this import at the top
from transcript import transcribe_video
import os
from ocr import ocr
import text_analysis as ta
import ast
import numpy as np
import models as m
import logging
logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
ad_df = pd.DataFrame(index=[0])  # Initialize DataFrame with one empty row
st.markdown("# Commercial Brand Differentiating Analysis Prediction Model")
st.markdown("## Input Data")
st.markdown("### Awareness Filters")
encoded_emotion = st.checkbox("Is the commercial emotional?", key="encoded_emotion_enabled")
encoded_emotion = 1 if encoded_emotion else 0

csr_type = st.checkbox("Are there any CSR (Corporate Social Responsibility) elements in the commercial?", key="csr_enabled")
csr_type = 1 if csr_type else 0
ad_df["encoded_emotion"] = encoded_emotion
ad_df["csr_type"] = csr_type
ad_df["commercial_number"] = 1
INDUSTRY_SPECIFIC_AWARENESS = st.checkbox("Enable Industry Knowledge", key="industry_enabled")
st.info('Our model knows of a select few Industries with Keywords commonly associated with a BDM. Enable this and check if your ad fits into one of these categories', icon="🔍")

product_cat_df = pd.read_csv("product_categories.csv")
product_category = st.selectbox(
"Select Product Category",
product_cat_df["product_cat_name"].unique(),
disabled=not st.session_state.industry_enabled,
)

if INDUSTRY_SPECIFIC_AWARENESS:
    product_cat_df = pd.read_csv("product_categories.csv")
    product_cat_keywords = product_cat_df[product_cat_df["product_cat_name"] == product_category]['product_cat_keywords']
    ad_df["product_cat_keywords"] = product_cat_keywords



BRAND_SPECIFIC_AWARENESS = st.checkbox("Enable Brand Knowledge", key="brand_enabled")
st.info('Our model knows of a select few Brands with Keywords commonly associated with a BDM. Enable this and check if your ad fits into one of these categories', icon="🔍")
product_brand_df= pd.read_csv("product_brands.csv")
product_brand = st.selectbox(
    "Select Brand",
    product_brand_df["brand"].unique(),
    disabled=not st.session_state.brand_enabled,
)

if BRAND_SPECIFIC_AWARENESS:
    product_brand_df = pd.read_csv("product_brands.csv")
    product_brand_keywords = product_brand_df[product_brand_df["brand"] == product_brand]['product_brand_keywords']
    
    ad_df["product_brand_keywords"] = product_brand_keywords.values[0]





st.markdown("###  Video Upload")
uploaded_file = st.file_uploader("Upload a Video of a Commercial to get started", type=["mp4"])
if uploaded_file is not None:
    # Read file as bytes
    bytes_data = uploaded_file.getvalue()
    st.write("File uploaded successfully!")

    # Save the uploaded file as an .mp4 file
    with open("uploaded_file.mp4", "wb") as f:
        f.write(bytes_data)
    st.write("File saved as uploaded_file.mp4")
    
    # Display the uploaded video
    st.video(uploaded_file)


st.markdown("## Output Data")

transcript = transcribe_video(f"{os.path.dirname(os.path.abspath(__file__))}/uploaded_file.mp4")

st.markdown("### Transcript")
ad_df["transcript"] = transcript
st.info('We have transcribe the following audio from the file you uploaded!', icon="🎙️")
st.markdown(f"> {transcript}")


st.markdown("### OCR")
st.info('The following words were detected in the frames of the file you uploaded!', icon="🎞️")
# INFO use the followingfor debugging
# ocr_text = "Your OCR text here"
logging.info(f"ad_df: {ad_df}")
ocr_text = ocr(f"{os.path.dirname(os.path.abspath(__file__))}/uploaded_file.mp4")
# ocr_text = 'THREE YEARS LATER AC Coming to Things Clydesdales Budweiser Chicago 7605 JV Kc CLYDESDALES RESPONSIBLY 02013 ANHEUSER BEER St Louis MO'
ad_df["ocr_text"] = ocr_text
st.markdown(f"> {ocr_text}")

# Add     ad_df = 
ad_df= ta.process_pronoun_data(ad_df, 'transcript')
ad_df = ta.process_pronoun_data(ad_df, 'ocr_text')
ad_df = ta.process_text_data(ad_df, 'transcript')
ad_df = ta.process_text_data(ad_df, 'ocr_text')
ad_df["transcript_adj_noun_pairs"] = ad_df["transcript"].apply(ta.extract_adj_noun_pairs)
ad_df["transcript_num_adj_noun_pairs"] = ad_df["transcript_adj_noun_pairs"].apply(len)
ad_df["ocr_text_adj_noun_pairs"] = ad_df["ocr_text"].apply(ta.extract_adj_noun_pairs)
ad_df["ocr_text_num_adj_noun_pairs"] = ad_df["ocr_text_adj_noun_pairs"].apply(len)

row = ad_df.iloc[0]
def display_results(row, text_column):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Words", row[f'{text_column}_word_count'])
    with col2:
        st.metric("BDM Terms", row[f'{text_column}_total_bdm_terms_count'])
    with col3:
        st.metric("BDM Terms %", f"{row[f'{text_column}_total_bdm_terms_pct']:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Superlatives**")
        st.markdown(f"Count: {row[f'{text_column}_superlative_count']} ({row[f'{text_column}_superlative_pct']:.1f}%)")
        st.markdown(f"> {', '.join(row[f'{text_column}_superlatives']) if row[f'{text_column}_superlatives'] else 'None found'}")
        
        st.markdown("**Comparatives**")
        st.markdown(f"Count: {row[f'{text_column}_comparative_count']} ({row[f'{text_column}_comparative_pct']:.1f}%)")
        st.markdown(f"> {''.join(row[f'{text_column}_comparatives']) if row[f'{text_column}_comparatives'] else 'None found'}")

    with col2:
        st.markdown("**Unique Words**")
        st.markdown(f"Count: {row[f'{text_column}_uniqueness_count']} ({row[f'{text_column}_uniqueness_pct']:.1f}%)")
        st.markdown(f"> {', '.join(row[f'{text_column}_unique_words']) if row[f'{text_column}_unique_words'] else 'None found'}")
        
        st.markdown("**Adjective-Noun Pairs**")
        st.markdown(f"Count: {row[f'{text_column}_num_adj_noun_pairs']}")
        st.markdown(f"> {', '.join(row[f'{text_column}_adj_noun_pairs']) if row[f'{text_column}_adj_noun_pairs'] else 'None found'}")
        
        st.markdown("**Most Common Pronoun**")
        st.markdown(f"Pronoun: {row[f'{text_column}_most_common_pronoun']}")
        st.markdown(f"Count: {row[f'{text_column}_most_common_pronoun_count']}")
        st.markdown(f"Percentage: {row[f'{text_column}_most_common_pronoun_pct']:.1f}%")

st.markdown("### Text Analysis")
st.info('Here is the detailed analysis of the commercial transcript!', icon="📊")
display_results(row, 'transcript')
st.markdown("### OCR Analysis")
st.info('Here is the detailed analysis of the OCR text!', icon="📊")
display_results(row, 'ocr_text')

def display_match(header, text, keywords, group):
    st.markdown(f"#### {header}")
    st.info(f'Here is the comparison of the {header} with the keywords!', icon="🔍")
    if group == 'product_cat':
        st.markdown(f"**Category**: {product_category}")
    elif group == 'product_brand':
        st.markdown(f"**Brand**: {product_brand}")
    st.markdown("**Keywords**:")
    st.code(', '.join(keyword.strip() for keyword in keywords))
    ad_df[f"{text}_product_cat_keywords"] = ', '.join(keywords)
    st.markdown("##### **Top Matching Keywords:**")
    s = ''
    for keyword in ad_df[f"{text}_{group}_keywords_top_keywords"].iloc[0].split(', '):
        s += "- " + keyword + "\n" 
    st.markdown(s)

    st.markdown("##### Average Semantic Similarity of top 3")
    st.metric('',f"{ad_df[f'{text}_{group}_keywords_similarity'].values[0]}")
    st.progress(ad_df[f'{text}_{group}_keywords_similarity'].values[0])

if INDUSTRY_SPECIFIC_AWARENESS:
    ad_df = ta.calculate_semantic_similarities(ad_df, 'transcript', 'product_cat_keywords')
    ad_df = ta.calculate_semantic_similarities(ad_df, 'ocr_text', 'product_cat_keywords')
    st.markdown("### Product Category Specificity")
    display_match('Transcript', 'transcript', product_cat_keywords, 'product_cat')
    display_match('OCR', 'ocr_text', product_cat_keywords, 'product_cat')
if BRAND_SPECIFIC_AWARENESS:

    ad_df = ta.calculate_semantic_similarities(ad_df, 'transcript', 'product_brand_keywords')
    ad_df = ta.calculate_semantic_similarities(ad_df, 'ocr_text', 'product_brand_keywords')
    st.markdown("### Product Brand Specificity")
    display_match('Transcript', 'transcript', product_brand_keywords, 'product_brand')
    display_match('OCR', 'ocr_text', product_brand_keywords, 'product_brand')

def beautify_df(df):
    for col in df.columns:
        st.write(f"**{col.capitalize()}** : {df.loc[0, col]}")




# Assuming ad_df is your DataFrame

# Sample DataFrame with existing columns
# Add the new row to the DataFrame
# Display the DataFrame
# Use pd.concat to append the new row
temp_df = ad_df.copy()
temp_df.rename(columns={
    'superlative_count': 'Superlative Count',
    'comparative_count': 'Comparative Count',
    'uniqueness_count': 'Uniqueness Count',
    'total_bdm_terms_count': 'Total BDM Terms Count',
    'total_bdm_terms_pct': 'Total BDM Terms Percentage',
    'num_adj_noun_pairs': 'Number of Adjective-Noun Pairs'
}, inplace=True)
# st.write(beautify_df(temp_df))
# add new row to ad_df with transcript and ocr text

ad_df.drop(columns=['commercial_number'], inplace=True)
ad_df = m.remove_unwanted_columns(ad_df)
trained_models = m.load_models(INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS)

# Continue with your data preparation
data = m.prepare_model_data(ad_df)
prediction = m.predict_model(data, trained_models)
majority_vote = prediction[['Logistic Regression_prediction', 'Random Forest_prediction', 'Support Vector Machine_prediction']].mode(axis=1).iloc[0, 0]
# results_df, predictions = m.evaluate_models(data, target, trained_models)

@st.dialog("The results of the model are as follows:")
def vote(majority_vote):
    if majority_vote == 1:
        st.success("✅ Our models predict that this commercial contains a strong BDM.")
    else:
        st.error("❌ Our models predict that this commercial does not contain a strong BDM.")
    st.write(prediction)
st.markdown("## Result")
if st.button("Click to see Result"):
    vote(majority_vote)
