from ocr import ocr
import os

ocr_text = ocr(f"{os.path.dirname(os.path.abspath(__file__))}/uploaded_file.mp4")
print(ocr_text)