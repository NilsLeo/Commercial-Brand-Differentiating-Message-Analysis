# Analysis Framework for Identifying Brand Differentiating Messages (BDM) in Commercials

<div style="text-align: center;">
  <img src="./Resources/images/SuperBowl.png" alt="Super Bowl" width="300"/>
</div>

This Project aims to provide a Framework which can be used to analyse a commercial and identify whether or not this commercial contains a Brand Differentiating Message (BDM)

---

# Installation/Usage

## Libraries

### Age and Gender Models

These models have been omitted from Version Control due to their filesize.Download them like so (Linux/ MAC OS)^[https://raw.githubusercontent.com/josemarcosrf/Age-Gender-Estimation-example/refs/heads/master/download_models.sh]:

```bash
cd "Final_Files/01. Bildanalyse/02. models"
mkdir "02. DEX"
cd "02. DEX"
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt
```

### Acoustic Indices Library

The Acoustic Indices Library can be added to this Project Like So:

```bash
cd "Final_Files/02. Tonanalyse/Acoustic_Indices"

git clone git@github.com:patriceguyot/Acoustic_Indices.git "./00 Quellcode"

```

### Face Classification Library

```bash
cd "Final_Files/01. Bildanalyse/02. models"

git clone git@github.com:oarriaga/face_classification.git "./03. emotion_model"
```

## Videos

The [Resources](./Resources) Directory should Contain the Raw Commercial Files from 2013 to 2022. In order to keep the filesize of this repo small, thees have been omitted and must be obtained separately and need to be placed in the folder structure manually like so:

```sh
Resources/
  └── Ads/
    └── ADs_IG_2013/
      ├── AD0252.mp4
      ├── AD0253.mp4
      ├── AD0254.mp4
      ├── ...
    └── ADs_IG_2014/
      ├── AD0301.mp4
      ├── AD0302.mp4
      ├── ...
```

This [Spreadsheet](./SB_AD_LIST__2013-2022.xlsx) contains some metadata and indicates whether or not the add contains a BDM based on human feedback.

## Python Dependencies

```bash
# create virtual environment
python -m venv venv
source venv/bin/activate
# Windows venv\Scripts\activate
pip install -r ./requirements.txt
```

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
