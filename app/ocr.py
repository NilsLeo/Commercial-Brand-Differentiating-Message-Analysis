import pandas as pd
import cv2
from easyocr import Reader
import matplotlib.pyplot as plt
import re
import enchant
import spacy
from rapidfuzz import fuzz
import enchant
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
import gc  # Import garbage collector interface


logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Wörterbuch für Englisch
english_dict = enchant.Dict("en_US")

# Lade spaCy-Modell
nlp = spacy.load("en_core_web_md")

# 1. OCR durchführen
def cover_watermark(frame, x, y, w, h, method="black"):
    if method == "black":
        frame[y:y+h, x:x+w] = 0  # Bereich mit Schwarz füllen
    return frame

def preprocess_image(frame, x, y, w, h, enhance_contrast=1.5, sharpen=True):
    #Verarbeitet ein Frame: Graustufen, Kontrastverstärkung, Schärfen und Wasserzeichen-Abdeckung.
    # Bild laden
    if frame is None:
        raise ValueError("Fehler: Übergebenes Frame ist 'None'.")
    # Bild in Graustufen umwandeln
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Graustufenbild in PIL-Format konvertieren
    pil_image = Image.fromarray(gray_frame)
    
    # Kontrast verstärken
    if enhance_contrast > 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(enhance_contrast)
    
    # Schärfen
    if sharpen:
        pil_image = pil_image.filter(ImageFilter.SHARPEN)
    
    # Zurück zu OpenCV-Format
    processed_frame  = np.array(pil_image)
    
    # Wasserzeichen abdecken
    processed_frame  = cover_watermark(processed_frame, x, y, w, h, method="black")
    
    return processed_frame 

def filter_results(results, min_height=24, min_confidence=0.9):
    filtered_results = []
    for bbox, text, confidence in results:
        # Höhe berechnen (Differenz der y-Koordinaten)
        height = abs(bbox[2][1] - bbox[0][1])  # untere y-Koordinate - obere y-Koordinate
        if height >= min_height and confidence >= min_confidence:
            filtered_results.append((bbox, text, confidence))
    return filtered_results

def process_frames_with_watermark_removal(frames, x, y, w, h, method="black"):
    """
    Entfernt Wasserzeichen und führt OCR auf allen Frames durch, gruppiert die Ergebnisse nach Frames.
    Args:
        frames (List[Dict]): Eine Liste von Dictionaries mit Frame-Pfad und Werbungsnamen.
        x, y (int): Obere linke Ecke des Wasserzeichen-Bereichs.
        w, h (int): Breite und Höhe des Wasserzeichen-Bereichs.
        method (str): Methode zum Abdecken des Wasserzeichens ("black" oder "blur").

    Returns:
        pd.DataFrame: Ein DataFrame mit den gruppierten OCR-Ergebnissen (Spalten: Frame, Recognized_Text).
    """
    results_list = []

    # Iteriere durch alle Frames
    for frame_idx, frame in enumerate(frames):
        # Abdeckung des Wasserzeichens
        processed_frame  = preprocess_image(frame, x, y, w, h, enhance_contrast=1.5, sharpen=True)

        # Perform text recognition with EasyOCR (nur auf das abgedeckte Bild)
        reader = Reader(['en'], gpu=True)
        if reader is not None:
            ocr_results = reader.readtext(processed_frame)
        else:
            raise ValueError("Ein EasyOCR-Reader muss übergeben werden!")

        # Filter OCR-Ergebnisse
        filtered_ocr_results = filter_results(ocr_results, min_height=24, min_confidence=0.4)

        # Ergebnisse speichern
        for (bbox, text, prob) in filtered_ocr_results:
            # Überprüfe, ob der Text gültig ist (nicht leer oder nur Sonderzeichen)
            if text and text.strip() and len(text.strip()) > 1:  # Sicherstellen, dass der Text sinnvoll ist
                results_list.append({
                    "Frame_index": frame_idx,  # Index des Frames in der Liste
                    "Recognized_Text": text.strip(),  # Direkter Text ohne zusätzliche Filterung
                    "Ad_Name": "AD",  # Werbung Name für alle Frames
                })

    # Ergebnisse in DataFrame umwandeln
    results_df = pd.DataFrame(results_list)

    # Gruppiere die Ergebnisse nach Frames
    merged_results = (
        results_df.groupby("Frame_index")
        .agg({
            "Recognized_Text": lambda x: ", ".join(x),  # Kombiniere alle erkannten Texte pro Frame
        })
        .reset_index()
    )

    return merged_results



# 2. Duplikate entfernen

# Enchant-Wörterbuch für Englisch
def count_correct_spelled_words(text):
    """Zählt, wie viele Wörter im Text laut Enchant korrekt geschrieben sind."""
    words = text.split()
    correct_count = 0
    for w in words:
        if english_dict.check(w):
            correct_count += 1
    return correct_count

# Funktion: Gruppiere Texte nach Ähnlichkeit und wähle den häufigsten
def remove_similar_texts(texts, threshold=60):
    text_groups = []
    # 1) Gruppiere nach Ähnlichkeit
    for text in texts:
        group_found = False
        for group in text_groups:
            if fuzz.ratio(text, group[0]) >= threshold:  # Überprüfe Ähnlichkeit
                group.append(text)
                group_found = True
                break
        if not group_found:
            text_groups.append([text])
    # 2) Wähle pro Gruppe den Text mit den meisten korrekt geschriebenen Wörtern
    #return [Counter(group).most_common(1)[0][0] for group in text_groups]
    best_texts = []
    for group in text_groups:
        # Für jeden Text in der Gruppe ermitteln wir die "Score" der korrekt geschriebenen Wörter
        best_text = max(group, key=lambda t: count_correct_spelled_words(t))
        best_texts.append(best_text)
    
    return best_texts


# Funktion: Sortiere die Frame-Spalte numerisch
def sort_frames(dataframe, frame_column="Frame"):
    """
    Sortiert die DataFrame basierend auf der AD-Nummer und der Frame-Nummer.
    Args:
        dataframe (pd.DataFrame): Die DataFrame mit der Frame-Spalte.
        frame_column (str): Der Name der Spalte, die sortiert werden soll.
    Returns:
        pd.DataFrame: Die sortierte DataFrame.
    """
    # Filtere Zeilen mit Texten, die weniger als 2 Zeichen haben
    dataframe = dataframe[dataframe["Recognized_Text"].str.len() >= 3].copy()

    # Konvertiere alle Werte der Spalte in Strings (falls nicht bereits)
    dataframe[frame_column] = dataframe[frame_column].astype(str)
    
    # Funktion zur Extraktion der AD-Nummer und der Frame-Nummer
    def extract_ad_and_frame(value):
        try:
            # Zerlege den Frame-Namen
            parts = value.split("_")
            # Extrahiere die Haupt-AD-Nummer
            ad_part = int(parts[0][2:])  # "AD0251" → 251
            # Extrahiere das Untersegment, wenn vorhanden
            sub_ad_part = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else 0
            # Extrahiere die Frame-Nummer
            frame_part = int(parts[2].split(".")[0])  # "Frame_250.png" → 250
            return (ad_part, sub_ad_part, frame_part)
        except (IndexError, ValueError):
            # Fehlerhafte Werte behalten die Haupt-AD-Nummer, aber kommen ans Ende innerhalb dieser Gruppe
            try:
                ad_part = int(value.split("_")[0][2:])  # Extrahiere nur die Haupt-AD-Nummer
                return (ad_part, float('inf'), float('inf'))
            except:
                # Wenn auch die Haupt-AD-Nummer nicht extrahiert werden kann, ans Ende schieben
                return (float('inf'), float('inf'), float('inf'))
    
    # Sortiere die DataFrame basierend auf AD-Nummer und Frame-Nummer
    dataframe = dataframe.sort_values(
        by=frame_column,
        key=lambda col: col.map(extract_ad_and_frame)
    )
    return dataframe

# Funktion: Erstelle die bereinigte Tabelle
def create_cleaned_dataframe(merged_results, text_column="Recognized_Text", threshold=60):
    # Bereinige und gruppiere die Texte
    unique_texts = remove_similar_texts(merged_results[text_column].tolist(), threshold=threshold)
    unique_texts = [text for text in unique_texts if text.strip()]  # Entferne leere oder nur Leerzeichen enthaltende Texte

    # Kombiniere die bereinigten Texte in eine einzige Zeichenkette
    combined_text = "; ".join(unique_texts)

    # Erstelle eine neue DataFrame
    cleaned_df = pd.DataFrame([{"Recognized_Text": combined_text}])

    return cleaned_df


def clean_recognized_text_with_spacy(text):
    """
    Bereinigt erkannte Texte und prüft Wörter auf ihre Gültigkeit basierend auf einem Wörterbuch (enchant)
    und Named Entity Recognition (NER) mit spaCy.
    
    Args:
        text (str): Eingabetext (eine lange Zeichenkette).
        
    Returns:
        str: Bereinigter Text.
    """
    # 1. Bindestriche durch Leerzeichen ersetzen. Sonderzeichen entfernen, nur alphanumerische Zeichen und Leerzeichen behalten
    # Ersetze Bindestriche durch Leerzeichen
    cleaned_text0 = re.sub(r"-", " ", text)
    # Entferne alle Sonderzeichen außer %, wenn es direkt nach einer Zahl steht
    cleaned_text1 = re.sub(r"[^\w\s%']|(?<=\s)'|'(?=\s)", "", cleaned_text0)
    # Entferne Apostrophe, die nicht zwischen zwei Buchstaben stehen
    cleaned_text2 = re.sub(r"(?<=\w)'(?!\w)", "", cleaned_text1)
    # Entferne % wenn es NICHT direkt hinter einer Zahl steht
    cleaned_text = re.sub(r"(?<!\d)%", "", cleaned_text2)

    # 2. Text in Wörter aufteilen und in Kleinbuchstaben umwandeln
    words_in_text = cleaned_text.split()
    #print("Aufgeteilte Wörter (Kleinschreibung):", words_in_text)

    # 3. spaCy Named Entity Recognition (NER)
    doc = nlp(cleaned_text)
    recognized_entities = {ent.text.lower() for ent in doc.ents}
    #print("Von spaCy erkannte Entitäten:", recognized_entities)

    # 4. Filtere Wörter basierend auf Wörterbuch und spaCy-Entitäten
    meaningful_words = [
        word for word in words_in_text
        if english_dict.check(word) or word.lower() in recognized_entities
    ]

    # 5. Bereinige den Text, indem nur relevante Wörter beibehalten werden
    final_cleaned_text = " ".join(meaningful_words)

    return final_cleaned_text


def extraction(video_path, kps):
    """
    Extrahiert Frames aus einem Video basierend auf einer definierten Keyframes-per-Second (KPS)-Rate
    und speichert sie im Arbeitsspeicher.

    Args:
        video_path (str): Pfad zur Videodatei.
        kps (int): Anzahl der Keyframes pro Sekunde.

    Returns:
        list: Liste von Frames (jeder Frame ist ein NumPy-Array).
    """
    # Überprüfen, ob die Video-Datei existiert
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Das Video {video_path} konnte nicht geöffnet werden.")

    # Video-Eigenschaften
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    hop = max(1, round(fps / kps))  # Sicherheitscheck für hop
    curr_frame = 0
    frames = []  # Liste, um die extrahierten Frames zu speichern

    # Frame-Extraktion
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video-Ende erreicht

        if curr_frame % hop == 0:
            frames.append(frame)  # Speichere Frame im Speicher

        curr_frame += 1

    # Ressourcen freigeben
    cap.release()

    #print(f"{len(frames)} Frames wurden erfolgreich extrahiert.")
    return frames

VIDEO_PATH = "uploaded_file.mp4"
KPS = 5  # Keyframes pro Sekunde


def process_all_ads(input_folder):
    """
    Hauptpipeline, die alle Schritte für die Verarbeitung aller Werbungen durchführt.
    Args:
        input_folder (str): Der Pfad zum Hauptordner mit den Frames.
    Returns:
        pd.DataFrame: Ein bereinigtes DataFrame mit allen Werbungen und kombinierten Texten.
    """
    # Frame extraction
    frames = extraction(input_folder, kps=3)
    # Log the number of ads found
    print(f"Number of frames found in {input_folder}: {len(frames)}")  # Logging the count of frames extracted

    # Führe OCR auf allen Frames durch und entfernt Wasserzeichen
    x, y, w, h=62, 545, 235, 60
    method="black"
    ocr_df = process_frames_with_watermark_removal(frames, x, y, w, h, method)

    #2 Bereinige und kombiniere die Texte
    final_ocr_df = create_cleaned_dataframe(ocr_df,text_column="Recognized_Text", threshold=50)
    #print('succesful grouped')
    # Zeige das bereinigte OCR-Ergebnis
    #print(" Finales OCR-Ergebnis nach Bereinigung:")
    #print(final_ocr_df.to_string(index=False))

    # Daten mit Wörterbuch abgleichen und 
    final_ocr_df["cleaned_text"] = final_ocr_df["Recognized_Text"].apply(clean_recognized_text_with_spacy)
    #print('succesful cleaned')
    
    # Alle Texte aus 'cleaned_text' werden als String ausgegeben
    cleaned_text = " ".join(final_ocr_df["cleaned_text"].dropna().tolist())
    #print("cleaned_text:", cleaned_text)
    logging.info(f"cleaned_text: {cleaned_text}")
    return cleaned_text


def ocr(video_path: str):
    # Attempt to clear VRAM by deleting large objects and collecting garbage
    gc.collect()
    text = process_all_ads(video_path)
    print("cleaned_text:", text)
    return text

