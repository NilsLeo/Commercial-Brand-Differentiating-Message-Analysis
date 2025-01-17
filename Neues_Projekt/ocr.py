import ocr_functions as of 

# Lade spaCy-Modell und englische Wörterbuch
#nlp = spacy.load("en_core_web_md")
#english_dict = enchant.Dict("en_US")

# Initialisiere EasyOCR mit der gewünschten Sprache (in dem Fall Englisch)
#reader = Reader(['en'], gpu=True)

# Frame extraction
video_path = "uploaded_file.mp4"

def ocr(video_path: str):
  text = of.process_all_ads(video_path)
  #text = print(text)
  #words ="Not Implemented Yet"
  # TODO: @Giulia implement ocr
  print("cleaned_text:", text)
  return text

ocr(video_path)

