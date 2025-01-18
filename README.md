# Analysis Framework for Identifying Brand Differentiating Messages (BDM) in Commercials
<center>
<div style="text-align: center;">
  <img src="./Resources/images/SuperBowl.png" alt="Super Bowl" width="300"/>
</div>
</center>
This University Project aims to provide a Framework which can be used to analyse a commercial and identify whether or not this commercial contains a Brand Differentiating Message (BDM). [This Spreadsheet](BDM.xlsx) contains a list of Superbowl Ads which are labeled with either 1 (BDM) or 0 (No BDM) by the marketing faculty of my University. Due to limitations in Data Quality (Subjectivity, Binary Classification) and Quantity (only a few hundred ads, class imbalance), this project should be considered as more of a conceptual approach, rather than a model with a high accuracy.

Due to the small number of Ads, we used a machine learning approach, with manually engineered features, rather than just passing text through to a nerual network.

---

# Prerequisites (for docker container later on)

- Python 3.9.12
- virtualenv `pip install virtualenv python-dotenv`

```bash
chmod +x ./setup.sh
sudo apt install libportaudio2 -y
sudo apt install python3-pyaudio -y
sudo apt install portaudio19-dev -y
sudo apt install python3-dev -y
sudo apt install python3-pip
sudo apt install -y libenchant-2-dev
pip install jupyter nbconvert
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md

```

- A hugging face api key in your bashrc or zshrc `export HF_API_KEY="hf....."`

# Installation/Usage

`streamlit run app.py --server.reload=True`

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

