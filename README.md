# Analysis Framework for Identifying Brand Differentiating Messages (BDM) in Commercials
<div style="text-align: center;">
  <img src="./Resources/images/SuperBowl.png" alt="Super Bowl" width="300"/>
</div>

This Project aims to provide a Framework which can be used to analyse a commercial and identify whether or not this commercial contains a Brand Differentiating Message (BDM)

---

# Installation/Usage

## Age and Gender Models

These models have been omitted from Version Control due to their filesize.Download them like so (Linux/ MAC OS)^[https://raw.githubusercontent.com/josemarcosrf/Age-Gender-Estimation-example/refs/heads/master/download_models.sh]:

```bash
cd "altes_projekt/01. Bildanalyse/02. models"
mkdir "02. DEX"
cd "02. DEX"
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt
```
## Acoustic Indices Library

The Acoustic Indices Library can be added to this Project Like So:
```bash
cd "altes_projekt/02. Tonanalyse/Acoustic_Indices"

git clone git@github.com:patriceguyot/Acoustic_Indices.git "./00 Quellcode"

```
## Face Classification Library
```bash
cd "altes_projekt/01. Bildanalyse/02. models"

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