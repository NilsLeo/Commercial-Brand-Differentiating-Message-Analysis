import whisper

# Initialize the model once at module level
model = whisper.load_model("tiny")  # Using base model as default - requires less memory

def transcribe_video(video_path: str) -> str:
    try:
        result = model.transcribe(video_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing video: {e}")
        return ""
