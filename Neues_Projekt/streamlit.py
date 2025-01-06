import streamlit as st
import pandas as pd  # Add this import at the top

# File uploader
uploaded_file = st.file_uploader("Choose a file")
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

# ... existing file uploader code ...

# Read the CSV file
ad_df = pd.read_csv("ad_df.csv")

# Get the first row
first_row = ad_df.iloc[0]


# csv columns: brand,commercial_number,BDM,transcript,audio_only_transcript,total_bdm_terms_pct,comparatives,superlatives,unique_words,bdm_words,product_cat_name,product_cat_keywords,product_cat_brands,keyword_similarity,similar_phrases,shared_keywords,shared_keywords_count

st.subheader("Brand")
st.write(first_row["brand"])
st.subheader("Product Category")
st.write(first_row["product_cat_name"])
st.subheader("BDM")
st.write(first_row["BDM"])
st.subheader("Transcript")
st.write(first_row["transcript"])
st.subheader("Audio Only Transcript")
st.write(first_row["audio_only_transcript"])
st.subheader("Total BDM Terms Percentage")
st.write(first_row["total_bdm_terms_pct"])
st.subheader("Comparatives")
st.write(first_row["comparatives"])
st.subheader("Superlatives")
st.write(first_row["superlatives"])
st.subheader("Unique Words")
st.write(first_row["unique_words"])
st.subheader("BDM Words")
st.write(first_row["bdm_words"])
st.subheader("Product Category Keywords")
st.write(first_row["product_cat_keywords"])
st.subheader("Product Category Brands")
st.write(first_row["product_cat_brands"])
st.subheader("Keyword Similarity")
st.write(first_row["keyword_similarity"])
st.subheader("Similar Phrases")
st.write(first_row["similar_phrases"])
st.subheader("Shared Keywords")
st.write(first_row["shared_keywords"])
st.subheader("Shared Keywords Count")
st.write(first_row["shared_keywords_count"])

