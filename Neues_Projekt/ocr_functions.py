from dotenv import load_dotenv
import os
import pandas as pd
import cv2
from easyocr import Reader
import matplotlib.pyplot as plt
import re
import enchant
import spacy

# Wörterbuch für Englisch
english_dict = enchant.Dict("en_US")

# Lade spaCy-Modell
nlp = spacy.load("en_core_web_md")


# 1. OCR durchführen
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
    for frame_info in frames:
        ad_name = frame_info["ad"]
        frame_path = frame_info["frame_path"]

        # Lade das Bild
        image = cv2.imread(frame_path)

        # Abdeckung des Wasserzeichens
        image = cover_watermark(image, x, y, w, h, method)

        # Perform text recognition with EasyOCR (nur auf das abgedeckte Bild)
        ocr_results = reader.readtext(image)

        # Ergebnisse speichern
        for (bbox, text, prob) in ocr_results:
            # Überprüfe, ob der Text gültig ist (nicht leer oder nur Sonderzeichen)
            if text and text.strip() and len(text.strip()) > 1:  # Sicherstellen, dass der Text sinnvoll ist
                results_list.append({
                    "Frame": os.path.basename(frame_path),  # Nur der Dateiname
                    "Recognized_Text": text.strip(),       # Direkter Text ohne zusätzliche Filterung
                })

    # Ergebnisse in DataFrame umwandeln
    results_df = pd.DataFrame(results_list)

    # Gruppiere die Ergebnisse nach Frames
    merged_results = (
        results_df.groupby("Frame")
        .agg({
            "Recognized_Text": lambda x: ", ".join(x),  # Kombiniere alle erkannten Texte pro Frame
        })
        .reset_index()
    )

    return merged_results


# 2. Duplikate entfernen

from collections import Counter
from rapidfuzz import fuzz
import enchant
from collections import Counter

# Enchant-Wörterbuch für Englisch
checker = enchant.Dict("en_US")
def count_correct_spelled_words(text):
    """Zählt, wie viele Wörter im Text laut Enchant korrekt geschrieben sind."""
    words = text.split()
    correct_count = 0
    for w in words:
        if checker.check(w):
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
def create_cleaned_dataframe(merged_results, ad_column="Frame", text_column="Recognized_Text", threshold=60):
    # Extrahiere den "Ad"-Namen aus den Frame-Namen
    merged_results["ad"] = merged_results[ad_column].apply(lambda x: x.split("_")[0])  
    # Gruppiere nach "ad" und bereinige Texte
    cleaned_data = []
    for ad, group in merged_results.groupby("ad"):
        unique_texts = remove_similar_texts(group[text_column].tolist(), threshold=threshold)
        unique_texts = [text for text in unique_texts if text.strip()]
        combined_text = "; ".join(unique_texts)  # Kombiniere die bereinigten Texte
        cleaned_data.append({"ad": ad, "recognized_text": combined_text})
    
    # Erstelle eine neue DataFrame
    cleaned_df = pd.DataFrame(cleaned_data)
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
    cleaned_text0 = re.sub(r"-", " ", text)
    cleaned_text1 = re.sub(r"[^\w\s']|(?<=\s)'|'(?=\s)", "", cleaned_text0)
    cleaned_text = re.sub(r"(?<=\w)'(?!\w)", "", cleaned_text1)
    print("Bereinigter Text ohne Sonderzeichen:", cleaned_text)

    # 2. Text in Wörter aufteilen und in Kleinbuchstaben umwandeln
    words_in_text = cleaned_text.split()
    print("Aufgeteilte Wörter (Kleinschreibung):", words_in_text)

    # 3. spaCy Named Entity Recognition (NER)
    doc = nlp(cleaned_text)
    recognized_entities = {ent.text.lower() for ent in doc.ents}
    print("Von spaCy erkannte Entitäten:", recognized_entities)

    # 4. Filtere Wörter basierend auf Wörterbuch und spaCy-Entitäten
    meaningful_words = [
        word for word in words_in_text
        if english_dict.check(word) or word.lower() in recognized_entities
    ]

    # 5. Bereinige den Text, indem nur relevante Wörter beibehalten werden
    final_cleaned_text = " ".join(meaningful_words)
    return final_cleaned_text


def process_all_ads(input_folder, x=62, y=545, w=235, h=60, method="black", threshold=70):
    """
    Hauptpipeline, die alle Schritte für die Verarbeitung aller Werbungen durchführt.
    Args:
        input_folder (str): Der Pfad zum Hauptordner mit den Frames.
    Returns:
        pd.DataFrame: Ein bereinigtes DataFrame mit allen Werbungen und kombinierten Texten.
    """
    # Lade alle Frames
    frames = load_all_frames(input_folder)
    
    # Führe OCR auf allen Frames durch und entfernt Wasserzeichen
    ocr_df = process_frames_with_watermark_removal(frames, x, y, w, h, method)
    
    # df frames sortieren
    sorted_ocr_df = sort_frames(ocr_df,"Frame")

    #2 Bereinige und kombiniere die Texte
    final_ocr_df = create_cleaned_dataframe(sorted_ocr_df,text_column="Recognized_Text", threshold=50)

    # Daten mit Wörterbuch abgleichen und 
    final_ocr_df["cleaned_text"] = final_ocr_df["recognized_text"].apply(clean_recognized_text_with_spacy)
    
    # Alle Texte aus 'cleaned_text' werden als String ausgegeben
    cleaned_text = " ".join(final_ocr_df["cleaned_text"].dropna().tolist())
        
    return cleaned_text