import streamlit as st
import pandas as pd  # Add this import at the top
from transcript import transcribe_video
import os
from ocr import ocr
import text_analysis as ta
import ast
import numpy as np
import models as m
# File uploader
st.markdown("# Commercial Brand Differentiating Analysis Prediction Model")
st.markdown("## Input Data")
product_brands_df = pd.read_csv("product_categories.csv")
product_category = st.selectbox(
    "Select Product Category",
product_brands_df["product_cat_name"].unique()
)


product_cat_keywords = product_brands_df[product_brands_df["product_cat_name"]==product_category]['product_cat_keywords'].values[0][1:-1].replace("'", "").split(", ")
# Add input fields
product_brand_df= pd.read_csv("product_brands.csv")


product_brand = st.selectbox(
    "Select Brand",
     product_brand_df["brand"].unique()
    )

product_brand_keywords = product_brand_df[product_brand_df["brand"]==product_brand]['product_brand_keywords'].values[0][1:-1].replace("'", "").split(", ")


ad_df = pd.DataFrame({
    "brand": [product_brand],
    "product_category": [product_category]
})

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
ad_df["audio_only_transcript"] = transcript
st.info('We have transcribe the following audio from the file you uploaded!', icon="🎙️")
st.markdown(f"> {transcript}")


st.markdown("### OCR")
st.info('The following words were detected in the frames of the file you uploaded!', icon="🎞️")
ocr_text = ocr(f"{os.path.dirname(os.path.abspath(__file__))}/uploaded_file.mp4")
st.markdown(f"> {ocr_text}")

# Add Text Analysis section
st.markdown("### Text Analysis")
st.info('Here is the detailed analysis of the commercial transcript!', icon="📊")

# Process text analysis for the single video
word_count = len(ta.get_tokens(transcript))
superlatives = ta.get_superlatives(transcript)
comparatives = ta.get_comparatives(transcript)
unique_words = ta.get_unique_words(transcript)
adj_noun_pairs = ta.extract_adj_noun_pairs(transcript)
# Calculate counts and percentages
superlative_count = len(superlatives)
comparative_count = len(comparatives)
uniqueness_count = len(unique_words)
num_adj_noun_pairs = len(adj_noun_pairs)

superlative_pct = (superlative_count / word_count * 100) if word_count > 0 else 0
comparative_pct = (comparative_count / word_count * 100) if word_count > 0 else 0
uniqueness_pct = (uniqueness_count / word_count * 100) if word_count > 0 else 0
total_bdm_terms = superlative_count + comparative_count + uniqueness_count
total_bdm_pct = (total_bdm_terms / word_count * 100) if word_count > 0 else 0

# Display results
st.markdown("#### Word Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Words", word_count)
with col2:
    st.metric("BDM Terms", total_bdm_terms)
with col3:
    st.metric("BDM Terms %", f"{total_bdm_pct:.1f}%")

st.markdown("#### Detailed Analysis")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Superlatives**")
    st.markdown(f"Count: {superlative_count} ({superlative_pct:.1f}%)")
    st.markdown(f"> {', '.join(superlatives) if superlatives else 'None found'}")
    
    st.markdown("**Comparatives**")
    st.markdown(f"Count: {comparative_count} ({comparative_pct:.1f}%)")
    st.markdown(f"> {', '.join(comparatives) if comparatives else 'None found'}")

with col2:
    st.markdown("**Unique Words**")
    st.markdown(f"Count: {uniqueness_count} ({uniqueness_pct:.1f}%)")
    st.markdown(f"> {', '.join(unique_words) if unique_words else 'None found'}")
    
    st.markdown("**Adjective-Noun Pairs**")
    st.markdown(f"Count: {num_adj_noun_pairs}")
    st.markdown(f"> {', '.join(adj_noun_pairs) if adj_noun_pairs else 'None found'}")

# Update the DataFrame with the analysis results
ad_df["superlatives"] = ', '.join(superlatives) if superlatives else ''
ad_df["comparatives"] = ', '.join(comparatives) if comparatives else ''
ad_df["unique_words"] = ', '.join(unique_words) if unique_words else ''
ad_df["total_bdm_terms_pct"] = total_bdm_pct

st.markdown("### Product Category Specificity")
st.info('Here is the comparison of the commercial transcript with the product category keywords!', icon="🔍")

st.markdown(f"**Category**: {product_category}")
st.markdown("**Keywords**:")
st.code(', '.join(keyword.strip() for keyword in product_cat_keywords))
ad_df["product_cat_keywords"] = ', '.join(product_cat_keywords)

# Calculate category similarities
product_cat_keyword_similarities = {
    keyword: round(float(ta.get_semantic_similarity(transcript, keyword)), 3)
    for keyword in product_cat_keywords
}

# Get top 3 category matches
cat_sorted_keywords = sorted(product_cat_keyword_similarities.items(), key=lambda x: x[1], reverse=True)
cat_top_3 = cat_sorted_keywords[:3]
cat_top_3_avg = round(float(np.mean([sim for _, sim in cat_top_3])), 3)

# Display category metrics
st.metric("Category Match Score", f"{cat_top_3_avg:.3f}")
st.markdown("**Top Matching Keywords (Average of the following top 3):**")
for keyword, similarity in cat_top_3:
    st.progress(similarity)
    st.caption(f"{keyword}: {similarity:.3f}")

st.markdown("### Brand Specificity")
st.info('Here is the comparison of the commercial transcript with the brand keywords!', icon="🔍")

st.markdown(f"**Brand**: {product_brand}")
st.markdown("**Keywords**:")
st.code(', '.join(keyword.strip() for keyword in product_brand_keywords))
ad_df["product_brand_keywords"] = ', '.join(product_brand_keywords)

# Calculate brand similarities
product_brand_keyword_similarities = {
    keyword: round(float(ta.get_semantic_similarity(transcript, keyword)), 3)
    for keyword in product_brand_keywords
}

# Get top 3 brand matches
brand_sorted_keywords = sorted(product_brand_keyword_similarities.items(), key=lambda x: x[1], reverse=True)
brand_top_3 = brand_sorted_keywords[:3]
brand_top_3_avg = round(float(np.mean([sim for _, sim in brand_top_3])), 3)

# Display brand metrics
st.metric("Brand Match Score (Average of the following top 3)", f"{brand_top_3_avg:.3f}")
st.markdown("**Top Matching Keywords:**")
for keyword, similarity in brand_top_3:
    st.progress(similarity)
    st.caption(f"{keyword}: {similarity:.3f}")

ad_df['product_cat_keyword_similarity'] = cat_top_3_avg
ad_df['product_cat_top_keywords'] = ', '.join([keyword for keyword, _ in cat_top_3])

ad_df['product_brand_keyword_similarity'] =brand_top_3_avg
ad_df['product_brand_top_keywords'] = ', '.join([keyword for keyword, _ in brand_top_3])


st.markdown("### Final Overview of all Data")
st.write(ad_df)

st.markdown("### Model Result")
trained_models = m.load_models()
data, target = m.prepare_model_data(ad_df)
results_df, predictions = m.evaluate_models(data, target, trained_models)

st.write(results_df)
