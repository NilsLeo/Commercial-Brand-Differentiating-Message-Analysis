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
st.markdown("# Super Bowl Commercial Brand Differentiating Analysis Prediction Model")
# center this image
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/1/16/Super_Bowl_logo.svg/1920px-Super_Bowl_logo.svg.png", width=200)


st.markdown("## Input Data")
st.markdown("### Awareness Filters")
encoded_emotion = st.radio(
    "Select the emotional appeal of the commercial:",
    ('Rational', 'Balanced', 'Emotional'),
    key="encoded_emotion_enabled"
)
# Map the selected options to numerical values
emotion_mapping = {'Rational': 1, 'Balanced': 2, 'Emotional': 3}
ad_df["rational"] = emotion_mapping[encoded_emotion] == 1
ad_df["balanced"] = emotion_mapping[encoded_emotion] == 2
ad_df["emotional"] = emotion_mapping[encoded_emotion] == 3


csr_type = st.checkbox("Are there any CSR (Corporate Social Responsibility) elements in the commercial?", key="csr_enabled")
csr_type = 1 if csr_type else 0
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
    ad_df["product_cat_keywords"] = product_cat_keywords.values[0]


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



    
# Display found pronouns on the frontend, differentiated by source
def display_results(row, text_column, header, INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Words", row[f'{text_column}_word_count'])
        st.metric("BDM Terms", row[f'{text_column}_total_bdm_terms_count'])
        st.metric("BDM Terms %", f"{row[f'{text_column}_total_bdm_terms_pct']:.1f}%")
    with col2:
        st.markdown("**Superlatives**")
        st.markdown(f"Count: {row[f'{text_column}_superlative_count']} ({row[f'{text_column}_superlative_pct']:.1f}%)")
        st.markdown(f"> {', '.join(row[f'{text_column}_superlatives']) if row[f'{text_column}_superlatives'] else 'None found'}")
        st.markdown("**Comparatives**")
        st.markdown(f"Count: {row[f'{text_column}_comparative_count']} ({row[f'{text_column}_comparative_pct']:.1f}%)")
        st.markdown(f"> {''.join(row[f'{text_column}_comparatives']) if row[f'{text_column}_comparatives'] else 'None found'}")
        st.markdown("**Comparisons**")
        st.markdown(f"Count: {row[f'{text_column}_num_comparisons']}")
        # Convert each tuple to a string before joining
        comparisons = ', '.join(map(str, row[f'{text_column}_comparisons'])) if row[f'{text_column}_comparisons'] else 'None found'
        st.markdown(f"> {comparisons}")
    with col3:
        st.markdown("**Unique Words**")
        st.markdown(f"Count: {row[f'{text_column}_uniqueness_count']} ({row[f'{text_column}_uniqueness_pct']:.1f}%)")
        st.markdown(f"> {', '.join(row[f'{text_column}_unique_words']) if row[f'{text_column}_unique_words'] else 'None found'}")
        st.markdown("**Adjective-Noun Pairs**")
        st.markdown(f"Count: {row[f'{text_column}_num_adj_noun_pairs']}")
        st.markdown(f"> {', '.join(row[f'{text_column}_adj_noun_pairs']) if row[f'{text_column}_adj_noun_pairs'] else 'None found'}")
    with col4:
        st.markdown("**Pronouns Found**")
        st.markdown("I" if row[f'{text_column}_contains_i'] else "")
        st.markdown("We" if row[f'{text_column}_contains_we'] else "")
        st.markdown("You" if row[f'{text_column}_contains_you'] else "")
        st.markdown("He" if row[f'{text_column}_contains_he'] else "")
        st.markdown("She" if row[f'{text_column}_contains_she'] else "")
        st.markdown("It" if row[f'{text_column}_contains_it'] else "")
        st.markdown("They" if row[f'{text_column}_contains_they'] else "")
        st.markdown("My" if row[f'{text_column}_contains_my'] else "")
        st.markdown("Our" if row[f'{text_column}_contains_our'] else "")
        st.markdown("Yours" if row[f'{text_column}_contains_yours'] else "")
        st.markdown("His" if row[f'{text_column}_contains_his'] else "")
        st.markdown("Her" if row[f'{text_column}_contains_her'] else "")
        st.markdown("Its" if row[f'{text_column}_contains_its'] else "")
        st.markdown("Their" if row[f'{text_column}_contains_their'] else "")
        st.markdown("Ours" if row[f'{text_column}_contains_ours'] else "")
        st.markdown("Your" if row[f'{text_column}_contains_your'] else "")
        
        

    if INDUSTRY_SPECIFIC_AWARENESS:
            st.markdown(f"#### Industry Match")
            st.info(f'Here is the comparison of the {header} with the keywords!', icon="🔍")
            st.markdown(f"**Category**: {product_category}")
            # replace ' and [ ] with ''
            all_keywords = ad_df[f"product_cat_keywords"].iloc[0]
            all_keywords = str(all_keywords)
            all_keywords = all_keywords.replace("'", "").replace("[", "").replace("]", "").split(', ')
            with st.expander("See All Keywords"):
                for keyword in all_keywords:
                    st.write(f"- {keyword}")
            st.markdown("##### **Top Matching Keywords:**")
            keywords = ad_df[f"{text_column}_product_cat_keywords_top_keywords"].iloc[0].split(', ')
            for keyword in keywords:
                st.write(f"- {keyword}")
            st.markdown("##### **Average Semantic Similarity of top 3**")
            st.metric('',f"{ad_df[f'{text_column}_product_cat_keywords_similarity'].values[0]}")
            st.progress(ad_df[f'{text_column}_product_cat_keywords_similarity'].values[0])


    if BRAND_SPECIFIC_AWARENESS:
            st.markdown(f"#### Brand Match")
            st.info(f'Here is the comparison of the {header} with the keywords!', icon="🔍")
            st.markdown(f"**Brand**: {product_brand}")
            # replace ' and [ ] with ''
            all_keywords = ad_df[f"product_brand_keywords"].iloc[0]
            all_keywords = str(all_keywords)
            all_keywords = all_keywords.replace("'", "").replace("[", "").replace("]", "").split(', ')
            with st.expander("See All Keywords"):
                for keyword in all_keywords:
                    st.write(f"- {keyword}")
            st.markdown("##### **Top Matching Keywords:**")
            keywords = ad_df[f"{text_column}_product_brand_keywords_top_keywords"].iloc[0].split(', ')
            for keyword in keywords:
                st.write(f"- {keyword}")
            st.markdown("##### **Average Semantic Similarity of top 3**")
            st.metric('',f"{ad_df[f'{text_column}_product_brand_keywords_similarity'].values[0]}")
            st.progress(ad_df[f'{text_column}_product_brand_keywords_similarity'].values[0])


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

########################################################

transcript = transcribe_video(f"{os.path.dirname(os.path.abspath(__file__))}/uploaded_file.mp4")
ad_df["transcript"] = transcript
ad_df["transcript_adj_noun_pairs"] = ad_df["transcript"].apply(ta.extract_adj_noun_pairs)
ad_df["transcript_num_adj_noun_pairs"] = ad_df["transcript_adj_noun_pairs"].apply(len)

ad_df['transcript_contains_i'] = ad_df['transcript'].apply(ta.contains_i)
ad_df['transcript_contains_we'] = ad_df['transcript'].apply(ta.contains_we)
ad_df['transcript_contains_you'] = ad_df['transcript'].apply(ta.contains_you)
ad_df['transcript_contains_he'] = ad_df['transcript'].apply(ta.contains_he)
ad_df['transcript_contains_she'] = ad_df['transcript'].apply(ta.contains_she)
ad_df['transcript_contains_it'] = ad_df['transcript'].apply(ta.contains_it)
ad_df['transcript_contains_they'] = ad_df['transcript'].apply(ta.contains_they)
ad_df['transcript_contains_us'] = ad_df['transcript'].apply(ta.contains_us)
ad_df['transcript_contains_them'] = ad_df['transcript'].apply(ta.contains_them)
ad_df['transcript_contains_my'] = ad_df['transcript'].apply(ta.contains_my)
ad_df['transcript_contains_our'] = ad_df['transcript'].apply(ta.contains_our)
ad_df['transcript_contains_ours'] = ad_df['transcript'].apply(ta.contains_ours)
ad_df['transcript_contains_your'] = ad_df['transcript'].apply(ta.contains_your)
ad_df['transcript_contains_yours'] = ad_df['transcript'].apply(ta.contains_yours)
ad_df['transcript_contains_his'] = ad_df['transcript'].apply(ta.contains_his)
ad_df['transcript_contains_her'] = ad_df['transcript'].apply(ta.contains_her)
ad_df['transcript_contains_its'] = ad_df['transcript'].apply(ta.contains_its)
ad_df['transcript_contains_their'] = ad_df['transcript'].apply(ta.contains_their)
ad_df['transcript_contains_theirs'] = ad_df['transcript'].apply(ta.contains_theirs)

ad_df["transcript_comparisons"] = ad_df["transcript"].apply(ta.apply_on_transcript)
ad_df["transcript_num_comparisons"] = ad_df["transcript_comparisons"].apply(len)

if INDUSTRY_SPECIFIC_AWARENESS:
    ad_df = ta.calculate_semantic_similarities(ad_df, 'transcript', 'product_cat_keywords')

if BRAND_SPECIFIC_AWARENESS:
    ad_df = ta.calculate_semantic_similarities(ad_df, 'transcript', 'product_brand_keywords')

ad_df = ta.process_text_data(ad_df, 'transcript')
st.markdown("## Transcript")
st.info('We have transcribe the following audio from the file you uploaded!', icon="🎙️")
st.markdown(f"> {transcript}")
row = ad_df.iloc[0]
display_results(row, 'transcript', 'Transcript', INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS)

########################################################

ocr_text = ocr(f"{os.path.dirname(os.path.abspath(__file__))}/uploaded_file.mp4")
# ocr_text = "WILL SOUND LIKE THIS 11500 pound FEET Torque PURE DOMINANCE quiet REVOLUTION IS COMING ZERO LIMITS production model shown Initial availability Fall 2021 SEE IT 52020"
ad_df["ocr_text"] = ocr_text
# ad_df = ta.process_pronoun_data(ad_df, 'ocr_text')
ad_df = ta.process_text_data(ad_df, 'ocr_text')
ad_df["ocr_text_adj_noun_pairs"] = ad_df["ocr_text"].apply(ta.extract_adj_noun_pairs)
ad_df["ocr_text_num_adj_noun_pairs"] = ad_df["ocr_text_adj_noun_pairs"].apply(len)

ad_df['ocr_text_contains_i'] = ad_df['ocr_text'].apply(ta.contains_i)
ad_df['ocr_text_contains_we'] = ad_df['ocr_text'].apply(ta.contains_we)
ad_df['ocr_text_contains_you'] = ad_df['ocr_text'].apply(ta.contains_you)
ad_df['ocr_text_contains_he'] = ad_df['ocr_text'].apply(ta.contains_he)
ad_df['ocr_text_contains_she'] = ad_df['ocr_text'].apply(ta.contains_she)
ad_df['ocr_text_contains_it'] = ad_df['ocr_text'].apply(ta.contains_it)
ad_df['ocr_text_contains_they'] = ad_df['ocr_text'].apply(ta.contains_they)
ad_df['ocr_text_contains_us'] = ad_df['ocr_text'].apply(ta.contains_us)
ad_df['ocr_text_contains_them'] = ad_df['ocr_text'].apply(ta.contains_them)
ad_df['ocr_text_contains_my'] = ad_df['ocr_text'].apply(ta.contains_my)
ad_df['ocr_text_contains_our'] = ad_df['ocr_text'].apply(ta.contains_our)
ad_df['ocr_text_contains_ours'] = ad_df['ocr_text'].apply(ta.contains_ours)
ad_df['ocr_text_contains_your'] = ad_df['ocr_text'].apply(ta.contains_your)
ad_df['ocr_text_contains_yours'] = ad_df['ocr_text'].apply(ta.contains_yours)
ad_df['ocr_text_contains_his'] = ad_df['ocr_text'].apply(ta.contains_his)
ad_df['ocr_text_contains_her'] = ad_df['ocr_text'].apply(ta.contains_her)
ad_df['ocr_text_contains_its'] = ad_df['ocr_text'].apply(ta.contains_its)
ad_df['ocr_text_contains_their'] = ad_df['ocr_text'].apply(ta.contains_their)
ad_df['ocr_text_contains_theirs'] = ad_df['ocr_text'].apply(ta.contains_theirs)

ad_df["ocr_text_comparisons"] = ad_df["ocr_text"].apply(ta.apply_on_transcript)
ad_df["ocr_text_num_comparisons"] = ad_df["ocr_text_comparisons"].apply(len)

if INDUSTRY_SPECIFIC_AWARENESS:

    ad_df = ta.calculate_semantic_similarities(ad_df, 'ocr_text', 'product_cat_keywords')

if BRAND_SPECIFIC_AWARENESS:
    ad_df = ta.calculate_semantic_similarities(ad_df, 'ocr_text', 'product_brand_keywords')


st.markdown("## OCR")
st.info('The following words were detected in the frames of the file you uploaded!', icon="🎞️")

row = ad_df.iloc[0]
st.markdown(f"> {ocr_text}")
display_results(row, 'ocr_text', 'OCR', INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS)



ad_df = m.prepare_df_for_modeling(ad_df)
ad_df.drop(columns=['commercial_number'], inplace=True)

trained_models = m.load_models(INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS)

# Continue with your data preparation
data = ad_df
prediction = m.predict_model(data, trained_models)
majority_vote = prediction.mode(axis=1).iloc[0, 0]
# name the row of the prediction dataframe
prediction.index = ['Prediction']
prediction.columns = prediction.columns.str.replace('_prediction', '')
prediction = prediction.replace(1, '✅').replace(0, '❌')
# results_df, predictions = m.evaluate_models(data, target, trained_models)

@st.dialog("Model Results")
def vote(majority_vote):
    if majority_vote == 1:
        st.success("The majority of our models predict that this commercial DOES contain a strong BDM.")
    else:
        st.error("The majority of our models predict that this commercial does NOT contain a strong BDM.")
    st.write(prediction.iloc[0])
st.markdown("## Result")
if st.button("Click to see Result"):
    vote(majority_vote)

