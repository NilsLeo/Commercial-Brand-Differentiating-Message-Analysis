from dotenv import load_dotenv
load_dotenv()

import subprocess
import os


urls = [    
"https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel",
"https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt",
"https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel",
"https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt"
]

# for url in urls:
#     subprocess.run(["wget", "-P", os.getenv("BILDANALYSE_MODELS_DEX_DIR"), url])
    
# subprocess.run(["git", "clone", "git@github.com:patriceguyot/Acoustic_Indices.git", os.getenv("TONANALYSE_ACOUSTIC_INDICES_QUELLCODE_DIR")])

# subprocess.run(["git", "clone", "git@github.com:oarriaga/face_classification.git", os.getenv("BILDANALYSE_MODELS_EMOTION_DIR")])

subprocess.run(["git", "clone", "git@github.com:x4nth055/gender-recognition-by-voice.git", os.getenv("TONANALYSE_AUDIO_GENDER_NOTEBOOKS_DIR")])

# subprocess.run(["wget", "https://box.fu-berlin.de/s/zwxKp8PXkCwAwGe/download"])

# subprocess.run(["unzip", os.getenv("ROOT_DIR")+ "/download", "-d", os.getenv("ROOT_DIR")])

# subprocess.run(["rm", os.getenv("ROOT_DIR")+ "/download"])