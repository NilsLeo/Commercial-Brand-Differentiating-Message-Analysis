import re
import enchant
import ocr_functions as of 
import spacy




# Lade spaCy-Modell und englische Wörterbuch
nlp = spacy.load("en_core_web_md")
english_dict = enchant.Dict("en_US")

# Initialisiere EasyOCR mit der gewünschten Sprache (in dem Fall Englisch)
reader = Reader(['en'], gpu=True)

def ocr(video_path: str):
  # Video path lesen
  print(video_path)
  # Video 
  text = of.process_all_ads(input_folder, x=62, y=545, w=235, h=60, method="black", threshold=70)
  words ="Not Implemented Yet"
  # TODO: @Giulia implement ocr
  return words