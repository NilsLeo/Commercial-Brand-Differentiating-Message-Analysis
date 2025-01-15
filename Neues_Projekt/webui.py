import streamlit as st
import pandas as pd  # Add this import at the top
from transcript import transcribe_video
import os
from ocr import ocr
import text_analysis as ta
# File uploader
st.markdown("# Commercial Brand Differentiating Analysis Prediction Model")
st.markdown("## Input Data")

uploaded_file = st.file_uploader("Upload a Video of a Commercial to get started", type=["mp4"])
if uploaded_file is not None:
    # Read file as bytes
    bytes_data = uploaded_file.getvalue()
    st.write("File uploaded successfully!")

    # Save the uploaded file as an .mp4 file
    with open("uploaded_file.mp4", "wb") as f:
        f.write(bytes_data)
    st.write("File saved as uploaded_file.mp4")


# Add input fields
product_brand = st.selectbox(
    "Select Brand",
     ["Snickers", "Avocados From Mexico","Other"]
    )

product_category = st.selectbox(
    "Select Product Category",
    ["Electronics", "Clothing", "Food", "Beauty", "Other"]
)
ad_df = pd.DataFrame({
    "brand": [product_brand],
    "product_category": [product_category]
})
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

# Calculate counts and percentages
superlative_count = len(superlatives)
comparative_count = len(comparatives)
uniqueness_count = len(unique_words)

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

# Update the DataFrame with the analysis results
ad_df["superlatives"] = ', '.join(superlatives) if superlatives else ''
ad_df["comparatives"] = ', '.join(comparatives) if comparatives else ''
ad_df["unique_words"] = ', '.join(unique_words) if unique_words else ''
ad_df["total_bdm_terms_pct"] = total_bdm_pct

st.markdown("### Final Overview of all Data")
st.write(ad_df)




# csv columns: brand,commercial_number,BDM,transcript,audio_only_transcript,total_bdm_terms_pct,comparatives,superlatives,unique_words,bdm_words,product_cat_name,product_cat_keywords,product_cat_brands,product_cat_keyword_similarity,similar_phrases,shared_keywords,shared_keywords_count

# st.subheader("Brand")
# st.write(first_row["brand"])
# st.subheader("Product Category")
# st.write(first_row["product_cat_name"])
# st.subheader("BDM")
# st.write(first_row["BDM"])
# st.subheader("Transcript")
# st.write(first_row["transcript"])
# st.subheader("Audio Only Transcript")
# transcribe_video(f"{os.path.abspath(__file__)}")
# st.write(first_row["audio_only_transcript"])

# st.subheader("Total BDM Terms Percentage")
# st.write(first_row["total_bdm_terms_pct"])
# st.subheader("Comparatives")
# st.write(first_row["comparatives"])
# st.subheader("Superlatives")
# st.write(first_row["superlatives"])
# st.subheader("Unique Words")
# st.write(first_row["unique_words"])
# st.subheader("BDM Words")
# st.write(first_row["bdm_words"])
# st.subheader("Product Category Keywords")
# st.write(first_row["product_cat_keywords"])
# st.subheader("Product Category Brands")
# st.write(first_row["product_cat_brands"])
# st.subheader("Keyword Similarity")
# st.write(first_row["product_cat_keyword_similarity"])
# st.subheader("Similar Phrases")
# st.write(first_row["similar_phrases"])
# st.subheader("Shared Keywords")
# st.write(first_row["shared_keywords"])
# st.subheader("Shared Keywords Count")
# st.write(first_row["shared_keywords_count"])

