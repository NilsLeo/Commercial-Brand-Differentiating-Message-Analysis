import os

# Define file path
env_file = '.env'

# Get current working directory
ROOT_DIR = os.getcwd()

# Define additional paths
input_frames_all = os.path.join(ROOT_DIR, '01_input_frames_all')

# Write to the .env file
with open(env_file, 'w') as f:
    f.write(f"HF_API_KEY={os.getenv("HF_API_KEY")}\n")
    f.write(f"PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512\n")
    f.write(f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n")
    f.write(f"PYTORCH_NO_CUDA_MEMORY_CACHING=1\n")



    
    f.write(f"ROOT_DIR={ROOT_DIR}\n")
    # 01_input_frames_all
    INPUT_FRAMES_ALL_DIR=os.path.join(ROOT_DIR, '01_input_frames_all')
    f.write(f"INPUT_FRAMES_ALL={INPUT_FRAMES_ALL_DIR}\n")
    # Final_Files
    FINAL_FILES_DIR=os.path.join(ROOT_DIR, 'Final_Files')
    f.write(f"FINAL_FILES={FINAL_FILES_DIR}\n")
   # Ads
    ADS_DIR=os.path.join(ROOT_DIR, 'ADs') 
    f.write(f"ADS_DIR={ADS_DIR}\n")
    
    ADS_WAV_DIR=os.path.join(ROOT_DIR, 'ADs_wav') 
    f.write(f"ADS_WAV_DIR={ADS_WAV_DIR}\n")
    # Final_Files/01. Bildanalyse
    BILDANALYSE_DIR=os.path.join(FINAL_FILES_DIR, '01. Bildanalyse')
    f.write(f"BILDANALYSE_DIR={BILDANALYSE_DIR}\n")
    # Final_Files/01. Bildanalyse/01. input_files
    BILDANALYSE_INPUT_FILES_DIR=os.path.join(BILDANALYSE_DIR, '01. input_files')
    f.write(f"BILDANALYSE_INPUT_FILES_DIR={BILDANALYSE_INPUT_FILES_DIR}\n")
    # Final_Files/01. Bildanalyse/02. models
    BILDANALYSE_MODELS_DIR=os.path.join(BILDANALYSE_DIR, '02. models')
    f.write(f"BILDANALYSE_MODELS_DIR={BILDANALYSE_MODELS_DIR}\n")
    # Final_Files/01. Bildanalyse/02. models/01. COCO_super_categories
    BILDANALYSE_MODELS_COCO_DIR=os.path.join(BILDANALYSE_MODELS_DIR, '01. COCO_super-categories')
    f.write(f"BILDANALYSE_MODELS_COCO_DIR={BILDANALYSE_MODELS_COCO_DIR}\n")
    # Final_Files/01. Bildanalyse/02. models/02. Dex
    BILDANALYSE_MODELS_DEX_DIR=os.path.join(BILDANALYSE_MODELS_DIR, '02. DEX')
    f.write(f"BILDANALYSE_MODELS_DEX_DIR={BILDANALYSE_MODELS_DEX_DIR}\n")
    # Final_Files/01. Bildanalyse/02. models/03. emotion_model
    BILDANALYSE_MODELS_EMOTION_DIR=os.path.join(BILDANALYSE_MODELS_DIR, '03. emotion_model')
    f.write(f"BILDANALYSE_MODELS_EMOTION_DIR={BILDANALYSE_MODELS_EMOTION_DIR}\n")
    # Final_Files/02. Tonanalyse
    TONANALYSE_DIR=os.path.join(FINAL_FILES_DIR, '02. Tonanalyse')
    f.write(f"TONANALYSE_DIR={TONANALYSE_DIR}\n")
    
    # Final_Files/02. Tonanalyse/Acoustic_Indices
    TONANALYSE_ACOUSTIC_INDICES_DIR=os.path.join(TONANALYSE_DIR, 'Acoustic_Indices')
    
    # Final_Files/02. Tonanalyse/splitted_audios
    TONANALYSE_SPLITTED_AUDIOS_DIR=os.path.join(TONANALYSE_DIR, 'splitted_audios')

    # Final_Files/02. Tonanalyse/audio_gender_notebooks
    TONANALYSE_AUDIO_GENDER_NOTEBOOKS_DIR=os.path.join(TONANALYSE_DIR, 'audio_gender_notebooks')
    f.write(f"TONANALYSE_AUDIO_GENDER_NOTEBOOKS_DIR={TONANALYSE_AUDIO_GENDER_NOTEBOOKS_DIR}\n")
    
    # Final_Files/02. Tonanalyse/Acoustic_Indices/00 Quellcode
    TONANALYSE_ACOUSTIC_INDICES_QUELLCODE_DIR=os.path.join(TONANALYSE_DIR, 'Acoustic_Indices', '00 Quellcode')
    f.write(f"TONANALYSE_ACOUSTIC_INDICES_QUELLCODE_DIR={TONANALYSE_ACOUSTIC_INDICES_QUELLCODE_DIR}\n")
    # Final_Files/03. Output Bild + Ton
    OUTPUT_BILD_PLUS_TON_DIR=os.path.join(FINAL_FILES_DIR, '03. Output Bild + Ton')
    f.write(f"OUTPUT_BILD_PLUS_TON_DIR={OUTPUT_BILD_PLUS_TON_DIR}\n")
    # Final_Files/03. Output Bild + Ton/01. output_lists
    OUTPUT_BILD_PLUS_TON_LISTS_DIR=os.path.join(OUTPUT_BILD_PLUS_TON_DIR, '01. output_lists')
    f.write(f"OUTPUT_BILD_PLUS_TON_LISTS_DIR={OUTPUT_BILD_PLUS_TON_LISTS_DIR}\n")
    # Final_Files/04. Ergebnisse
    ERGEBNISSE_DIR=os.path.join(FINAL_FILES_DIR, '04. Ergebnisse')
    f.write(f"ERGEBNISSE_DIR={ERGEBNISSE_DIR}\n")
    
    # Final_Files/02. Tonanalyse/splitted_audios
    TONANALYSE_SPLITTED_AUDIOS_DIR=os.path.join(TONANALYSE_DIR, 'Geschlechtserkennung','splitted_audios')
    f.write(f"TONANALYSE_SPLITTED_AUDIOS_DIR={TONANALYSE_SPLITTED_AUDIOS_DIR}\n")
