# Use the official Python image as the base
FROM python:3.9.12

# Install necessary utilities for Streamlit and Jupyter
RUN apt update -y && apt install -y \
  curl \
  wget \
  libportaudio2 \
  python3-pyaudio \
  portaudio19-dev \
  python3-dev \
  python3-pip \
  ffmpeg \
  libenchant-2-dev \
  libopencv-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN apt update -y && apt install libopencv-dev ffmpeg -y && pip install opencv-python

COPY ./app/.streamlit/config.toml /root/.streamlit/
COPY ./app/requirements.txt .
# Upgrade pip
RUN /usr/local/bin/python -m pip install --upgrade pip

# Install the dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Install additional Python packages
RUN pip install jupyter nbconvert opencv-python
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md

# Set the working directory inside the container
WORKDIR /app

# Create a virtual environment at the root of the working directory
RUN python -m venv .venv
RUN /bin/bash -c "source .venv/bin/activate"


# Expose the ports for Streamlit (8501) and Jupyter Notebook (8888)
EXPOSE 8501 8888

# Command to run both Streamlit and Jupyter in the background
CMD bash -c "source .venv/bin/activate && streamlit run '6 Deployment.py' & jupyter notebook --ip='0.0.0.0' --port=8888 --allow-root --NotebookApp.token=''"
