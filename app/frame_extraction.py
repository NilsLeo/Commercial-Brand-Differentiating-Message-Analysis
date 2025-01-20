import cv2


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

