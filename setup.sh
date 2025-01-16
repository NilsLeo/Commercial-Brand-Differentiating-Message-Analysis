#!/bin/bash

generate_env_vars() {
    # Define file path
    local env_file=".env"

    # Get current working directory
    local ROOT_DIR=$(pwd)

    # Write to the .env file
    {
        echo "HF_API_KEY=${HF_API_KEY}"
        echo "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
        echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        echo "PYTORCH_NO_CUDA_MEMORY_CACHING=1"
        
        echo "ROOT_DIR=${ROOT_DIR}"
        declare -a years=("2013" "2014") 
        echo "YEARS=${years[@]}"
        # 01_input_frames_all
        local INPUT_FRAMES_ALL_DIR="${ROOT_DIR}/01_input_frames_all"
        echo "INPUT_FRAMES_ALL=${INPUT_FRAMES_ALL_DIR}"
        
        # Final_Files
        local FINAL_FILES_DIR="${ROOT_DIR}/Final_Files"
        echo "FINAL_FILES=${FINAL_FILES_DIR}"
        
        # Ads
        local ADS_DIR="${ROOT_DIR}/ADs"
        echo "ADS_DIR=${ADS_DIR}"
        
        # local ADS_WAV_DIR="${ROOT_DIR}/ADs_wav"
        # echo "ADS_WAV_DIR=${ADS_WAV_DIR}"
    
        local BDM_EXCEL_FILE="${ADS_DIR}/SB_AD_LIST__2013-2022.xlsx"
        echo "BDM_EXCEL_FILE=${BDM_EXCEL_FILE}"


        # Final_Files/01. Bildanalyse
        local BILDANALYSE_DIR="${FINAL_FILES_DIR}/01. Bildanalyse"
        echo "BILDANALYSE_DIR=${BILDANALYSE_DIR}"
        
        # Final_Files/01. Bildanalyse/01. input_files
        local BILDANALYSE_INPUT_FILES_DIR="${BILDANALYSE_DIR}/01. input_files"
        echo "BILDANALYSE_INPUT_FILES_DIR=${BILDANALYSE_INPUT_FILES_DIR}"
        
        # Final_Files/01. Bildanalyse/02. models
        local BILDANALYSE_MODELS_DIR="${BILDANALYSE_DIR}/02. models"
        echo "BILDANALYSE_MODELS_DIR=${BILDANALYSE_MODELS_DIR}"
        
        # Final_Files/01. Bildanalyse/02. models/01. COCO_super_categories
        local BILDANALYSE_MODELS_COCO_DIR="${BILDANALYSE_MODELS_DIR}/01. COCO_super-categories"
        echo "BILDANALYSE_MODELS_COCO_DIR=${BILDANALYSE_MODELS_COCO_DIR}"
        
        # Final_Files/01. Bildanalyse/02. models/02. Dex
        local BILDANALYSE_MODELS_DEX_DIR="${BILDANALYSE_MODELS_DIR}/02. DEX"
        echo "BILDANALYSE_MODELS_DEX_DIR=${BILDANALYSE_MODELS_DEX_DIR}"
        
        # Final_Files/01. Bildanalyse/02. models/03. emotion_model
        local BILDANALYSE_MODELS_EMOTION_DIR="${BILDANALYSE_MODELS_DIR}/03. emotion_model"
        echo "BILDANALYSE_MODELS_EMOTION_DIR=${BILDANALYSE_MODELS_EMOTION_DIR}"
        
        # Final_Files/02. Tonanalyse
        local TONANALYSE_DIR="${FINAL_FILES_DIR}/02. Tonanalyse"
        echo "TONANALYSE_DIR=${TONANALYSE_DIR}"
        
        # Final_Files/02. Tonanalyse/Acoustic_Indices
        local TONANALYSE_ACOUSTIC_INDICES_DIR="${TONANALYSE_DIR}/Acoustic_Indices"
        
        # Final_Files/02. Tonanalyse/splitted_audios
        local TONANALYSE_SPLITTED_AUDIOS_DIR="${TONANALYSE_DIR}/splitted_audios"
        
        # Final_Files/02. Tonanalyse/audio_gender_notebooks
        local TONANALYSE_AUDIO_GENDER_NOTEBOOKS_DIR="${TONANALYSE_DIR}/audio_gender_notebooks"
        echo "TONANALYSE_AUDIO_GENDER_NOTEBOOKS_DIR=${TONANALYSE_AUDIO_GENDER_NOTEBOOKS_DIR}"
        
        # Final_Files/02. Tonanalyse/Acoustic_Indices/00 Quellcode
        local TONANALYSE_ACOUSTIC_INDICES_QUELLCODE_DIR="${TONANALYSE_DIR}/Acoustic_Indices/00 Quellcode"
        echo "TONANALYSE_ACOUSTIC_INDICES_QUELLCODE_DIR=${TONANALYSE_ACOUSTIC_INDICES_QUELLCODE_DIR}"
        
        # Final_Files/03. Output Bild + Ton
        local OUTPUT_BILD_PLUS_TON_DIR="${FINAL_FILES_DIR}/03. Output Bild + Ton"
        echo "OUTPUT_BILD_PLUS_TON_DIR=${OUTPUT_BILD_PLUS_TON_DIR}"
        
        # Final_Files/03. Output Bild + Ton/01. output_lists
        local OUTPUT_BILD_PLUS_TON_LISTS_DIR="${OUTPUT_BILD_PLUS_TON_DIR}/01. output_lists"
        echo "OUTPUT_BILD_PLUS_TON_LISTS_DIR=${OUTPUT_BILD_PLUS_TON_LISTS_DIR}"
        
        local FINAL_EXCEL_FILE="${OUTPUT_BILD_PLUS_TON_DIR}/End_Datei.xlsx"
        echo "FINAL_EXCEL_FILE=${FINAL_EXCEL_FILE}"

        # Final_Files/04. Ergebnisse
        local ERGEBNISSE_DIR="${FINAL_FILES_DIR}/04. Ergebnisse"
        echo "ERGEBNISSE_DIR=${ERGEBNISSE_DIR}"
        
        # Final_Files/02. Tonanalyse/splitted_audios
        local TONANALYSE_SPLITTED_AUDIOS_DIR="${TONANALYSE_DIR}/Geschlechtserkennung/splitted_audios"
        echo "TONANALYSE_SPLITTED_AUDIOS_DIR=${TONANALYSE_SPLITTED_AUDIOS_DIR}"
    } > "$env_file"
}

download_files() {
    set -a
    source .env
    set +a
    # Download files from URLs
    local urls=(
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel"
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt"
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel"
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt"
    )
    for url in "${urls[@]}"; do
        wget -P "$BILDANALYSE_MODELS_DEX_DIR" "$url"
    done

    # Clone repositories
    # git clone "git@github.com:patriceguyot/Acoustic_Indices.git" "$TONANALYSE_ACOUSTIC_INDICES_QUELLCODE_DIR"
    # git clone "git@github.com:oarriaga/face_classification.git" "$BILDANALYSE_MODELS_EMOTION_DIR"
    # git clone "git@github.com:x4nth055/gender-recognition-by-voice.git" "$TONANALYSE_AUDIO_GENDER_NOTEBOOKS_DIR"
    # git clone "git@github.com:patriceguyot/Acoustic_Indices.git" "$TONANALYSE_ACOUSTIC_INDICES_QUELLCODE_DIR"

    # # Download, unzip, and remove file
    # local download_url="https://box.fu-berlin.de/s/zwxKp8PXkCwAwGe/download"
    # local download_path="$ROOT_DIR/download"
    
    # wget -O "$download_path" "$download_url"
    # unzip "$download_path" -d "$ROOT_DIR"
    # rm "$download_path"
}

reduce_ads_selection() {
    for subfolder in "$ADS_DIR"/*; do
        if [ -d "$subfolder" ]; then
            ls "$subfolder" | tail -n +3 | while read -r file; do
                rm -rf "$subfolder/$file"
            done
        fi
    done
}

delete_gitignore_files() {
    declare -a gitignore_files
    while IFS= read -r pattern; do
        gitignore_files+=(\""./$pattern\"")
    done < .gitignore
    for file in "${gitignore_files[@]}"; do
        echo "Deleting $file"
        rm -rf "$file"
    done
}

setup_and_execute() {
    local env_path=$1
    local base_name=$2
    local venv_name="${base_name}-venv"
    
    cp .env "$env_path"
    cd "$env_path" || exit
        python -m venv "$venv_name"
    source "./${venv_name}/bin/activate"
    
    # Install jupyter and nbconvert explicitly
    pip install jupyter nbconvert
    pip install -r "${base_name}.requirements.txt"
    
    jupyter-nbconvert --to notebook --execute "${base_name}.ipynb" --inplace
    deactivate
    cd "$ROOT_DIR" || exit
}
convert_mp4_to_wav() {
    # Find all MP4 files in the current directory and subdirectories
    find . -type f -name "*.mp4" | while read -r video; do
        # Create WAV filename by replacing .mp4 with .wav
        wav_file="./${video%.mp4}.wav"
        echo "Converting: $video to $wav_file"
        # Added -y flag at the beginning to force overwrite
        ffmpeg -y -i "./$video" -vn -acodec pcm_s16le -ar 44100 -ac 2 "$wav_file"

    done
}

main_workflow() {
    # Uncomment generate_env_vars to ensure .env file exists
    # generate_env_vars
    # delete_gitignore_files
    download_files

    # reduce_ads_selection
    # convert_mp4_to_wav
    # setup_and_execute "./Final_Files/01. Bildanalyse/03. main_Script/" "03. main_Bildanalyse"
    # setup_and_execute "./Final_Files/02. Tonanalyse/main_sound_recognition_FINAL/" "main_sound_recognition_FINAL"
    # setup_and_execute "./Final_Files/03. Output Bild + Ton/02. Final Excel File/ End_Datei_Code/" "End_Datei_Code"
    # setup_and_execute "./Neues_Projekt/" "BDM_Detection"
}

main_workflow