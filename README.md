# Analysis Framework for Identifying Brand Differentiating Messages (BDM) in Commercials
<center>
<div style="text-align: center;">
  <img src="./Resources/images/SuperBowl.png" alt="Super Bowl" width="300"/>
</div>
</center>
This Project aims to provide a Framework which can be used to analyse a commercial and identify whether or not this commercial contains a Brand Differentiating Message (BDM)

---

# Prerequisites (for docker container later on)

- python
- virtualenv `pip install virtualenv python-dotenv`

```bash
chmod +x ./setup.sh
sudo apt install libportaudio2 -y
sudo apt install python3-pyaudio -y
sudo apt install portaudio19-dev -y
sudo apt install python3-dev -y
sudo apt install python3-pip
pip install jupyter nbconvert
```

- A hugging face api key in your bashrc or zshrc `export HF_API_KEY="hf....."`

# Installation/Usage



This allows for an environment with separate dependencies and packages from Host OS. Execute this everytime you make changes to the project.

The codebase was developed on the Following System

```bash
OS: Ubuntu 24.04.1 LTS x86_64 
Host: MS-7E07 1.0 
Kernel: 6.8.0-48-generic
Shell: zsh 5.9 
CPU: 13th Gen Intel i7-13700K (24) @ 5.500GHz 
GPU: Intel Raptor Lake-S GT1 [UHD Graphics 770] 
NVIDIA GeForce RTX 4070 
Memory: 7400MiB / 64042MiB 
```

The System-specific dependencies are as follows and may need to be installed differently based the systems specs, especially when using a different gpu (or lack thereof)

```bash
pip install torch torchvision torchaudio
pip install tensorflow\[and-cuda\]
```

The remaining dependencies should be installable irrespective of Operating System or hardware


```bash
pip install -r ./requirements.txt
```

Execute all of the following notebooks in the following order. If there are no Errors, all necessary dependencies are installed and the project works

- [Bildanalyse MAIN Script](./Final_Files/01.%20Bildanalyse/03.%20main_Script/03.%20main_Bildanalyse%20copy.ipynb)
- [Heatmap_Bildkomposition.ipynb](./Final_Files/01.%20Bildanalyse/05.%20Heatmaps_Bildkomposition/Heatmap_Bildkomposition.ipynb)
- [Manuelle Prüfung der Indices](./Final_Files/02.%20Tonanalyse/Acoustic_Indices/01%20Manueller%20Vergleich/00%20Manuelle%20Prüfung%20der%20Indices.ipynb)
- [Tonanalyse MAIN Script](./Final_Files/02.%20Tonanalyse/main_sound_recognition_FINAL.ipynb)
- [End Datei Code](./Final_Files/03.%20Output%20Bild%20+%20Ton/02.%20Final%20Excel%20File/End_Datei_Code.ipynb)
- [Identifikation der Attribute](./Final_Files/04.%20Ergebnisse/04.01.%20Identifikation%20der%20Attribute.ipynb)
- [Beste Werbespots](./Final_Files/04.%20Ergebnisse/04.02.%20Beste%20Werbespots.ipynb) 

### Komplettes Skript

Run [this Script](./setup.sh) to install and run the entire thing

# TODOS
- [ ] You need to refactor the code in such a way that it simply receives an ad and a brand and completes all steps. This is the groundwork for the eventual full-stack project
- [ ] Get rid of `!pip install` ans `%pip install`
- [ ] clone the git repos in download_gitignore_files.py instead of jupyter notebooks
- [ ] pip freeze once verything works
- [ ] speed up testing by only keeping one add per year initially
- [ ] write everything to one excel per ad vs. the original code which has multiple excel files for same ad
- add folder 2024, place ad inside, and test it


---

# Repository Structure

## Last Semester's Project

### Final Files Directory Description

#### [Image Analysis](./Final_Files/01.%20Bildanalyse/)

1. [input_frames](./input_frames): Examples of frames generated from the commercials. All other frames can be found in the directory [01_input_frames_all](./01_input_frames_all).
2. [models](./Final_Files/01.%20Bildanalyse/02.%20models/): Required models for image analysis.
3. [main_Script](./Final_Files/01.%20Bildanalyse/03.%20main_Script/03.%20main_Bildanalyse.ipynb): Comprehensive Python code for image analysis.
4. [output_frames](./Final_Files/01.%20Bildanalyse/): Sample output images after analysis.
5. [Heatmaps_Bildkomposition](./Final_Files/01.%20Bildanalyse/05.%20Heatmaps_Bildkomposition/Heatmap_Bildkomposition.ipynb): Python code for generating heatmaps.
6. [Manual Evaluation](./Final_Files/01.%20Bildanalyse/06.%20Manuelle%20Evaluation/): Model evaluation.

#### [Audio Analysis](./Final_Files/02.%20Tonanalyse/)

1. [Acoustic_Indices](./Final_Files/02.%20Tonanalyse/Acoustic_Indices/): Source code and manual evaluation.
2. [Gender Detection](./Final_Files/02.%20Tonanalyse/Geschlechtserkennung/): Generated audio segments and manual evaluation.
3. [Sentiment Analysis](./Final_Files/02.%20Tonanalyse/Stimmungsanalyse/): WhisperAI files and manual evaluation.
4. [main_sound_recognition_FINAL](./Final_Files/02.%20Tonanalyse/main_sound_recognition_FINAL.ipynb): Comprehensive Python code for audio analysis.

#### [Output Image+Audio](./Final_Files/03.%20Output%20Bild%20+%20Ton/)

1. [output_lists](./Final_Files/03.%20Output%20Bild%20+%20Ton/01.%20output_lists/): Excel lists for each commercial.
2. [Final Excel File](./Final_Files/03.%20Output%20Bild%20+%20Ton/02.%20Final%20Excel%20File/): Holistic Excel file for all commercials.

#### [Results](./Final_Files/04.%20Ergebnisse/)

1. [Attribute Identification](./Final_Files/04.%20Ergebnisse/04.01.%20Identifikation%20der%20Attribute.ipynb): Python code for identifying key attributes for evaluation.
2. [Best Commercials](./Final_Files/04.%20Ergebnisse/04.02.%20Beste%20Werbespots.ipynb): Python code for comparing the highest-rated commercial with others.
