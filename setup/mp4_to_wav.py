import os
import ffmpeg
from dotenv import load_dotenv

load_dotenv()

def convert_mp4_to_wav(src_folder, dest_folder):
    # Walk through all files and folders in the source folder
    for root, _, files in os.walk(src_folder):
        # Construct the target directory by replacing the source folder with the destination folder
        target_dir = root.replace(src_folder, dest_folder)

        # Ensure the corresponding target directory exists
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            # Process only mp4 files
            if file.endswith(".mp4"):
                # Define full path for the source and destination files
                src_path = os.path.join(root, file)
                dest_path = os.path.join(target_dir, os.path.splitext(file)[0] + ".wav")

                # Convert mp4 to wav
                try:
                    ffmpeg.input(src_path).output(dest_path).run()
                    print(f"Converted: {src_path} -> {dest_path}")
                except ffmpeg.Error as e:
                    print(f"Error converting {src_path}: {e}")
os.makedirs(os.getenv("ADS_WAV_DIR"), exist_ok=True)
convert_mp4_to_wav(os.getenv("ADS_DIR"), os.getenv("ADS_WAV_DIR"))