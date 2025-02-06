# Analysis Framework for Identifying Brand Differentiating Messages (BDM) in Commercials

<div style="text-align: center;">
  <img src="./images/SuperBowl.png" alt="Super Bowl" width="300"/>
</div>

This University Project aims to provide a Framework which can be used to analyse a commercial and identify whether or not this commercial contains a Brand Differentiating Message (BDM). [This Spreadsheet](./app/BDM.xlsx) contains a list of Superbowl Ads which are labeled with either 1 (BDM) or 0 (No BDM) by the marketing faculty of my University. Due to limitations in Data Quality (Subjectivity, Binary Classification) and Quantity (only a few hundred ads, class imbalance), this project should be considered as more of a conceptual approach, rather than a model with a high accuracy.

Due to the small number of Ads, we used a machine learning approach, with manually engineered features, rather than just passing text through to a neural network.

## CRISP-DM

This Project is structured using the CRISP-DM Framework

### Data Understanding
[Jupyter Notebook](<./app/2 - Data Understanding.ipynb>)
### Data Preparation, Modeling, Evaluation
[Jupyter Notebook](<./app/3-5 Data Preparation, Modeling, Evaluation.ipynb>)
### Deployment
[Webapp](<./app/6 Deployment.py>)

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Demo Application](#demo-application)
  - [Development](#development)
    - [Jupyter](#jupyter)
    - [Other Editors](#other-editors)
    - [Tips](#tips)
- [Contributing](#contributing)
- [License](#license)

## Getting Started
https://github.com/user-attachments/assets/d8b9f198-974f-4928-a6dc-fa378379d23c

### Prerequisites
- Docker
- docker-compose
- (Model Training Only) A URL with the Videos referenced in the model training. This must be added to your local .env file. Copy the [example](./app/.env.example) and adjust the URL

#### System Requirements
The model was developed on the following System

| Component  | Details                          |
|------------|----------------------------------|
| **OS**     | Ubuntu 24.04.1 LTS x86_64       |
| **Host**   | MS-7E07 1.0                     |
| **CPU**    | 13th Gen Intel i7-13700K (24) @ 5.500GHz |
| **GPU**    | NVIDIA GeForce RTX 4070         |
| **Memory** | 64042 MiB                       |

In in attempt to ensure cross plattform compatibility, a Docker image was created. However this has only been tested on 1 machine and needs to be tested on a broad spectrum of systems to ensure true Cross Plattform Compatibility

### Installation
To install, simply run the following command:
```bash
docker-compose up -d
```

## Usage
### Demo Application
Open [http://localhost:8502](http://localhost:8502) to access the Demo Web UI. This will allow you to upload a Video and see the steps that our Model goes through before giving its prediction whether or not the Video contains a strong BDM or not.

### Development
The source code is directly mapped to the [app](./app/) directory. This means that any changes made to the directory inside or outside of the container will be synced both ways.

#### Jupyter 
Open [http://localhost:8889](http://localhost:8889) to access the integrated Jupyter Development environment. The necessary dependencies are already installed.

In order to authenticate and set a password, in your terminal run the following commands:

```bash
docker exec -it app sh
jupyter server list
```
This will retrieve the token.
#### Tips
If you want to persist the changes, make sure to update the requirements.txt when it comes to python packages and any external dependencies must be added to the [Dockerfile](./Dockerfile).