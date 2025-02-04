# Analysis Framework for Identifying Brand Differentiating Messages (BDM) in Commercials

<div style="text-align: center;">
  <img src="./images/SuperBowl.png" alt="Super Bowl" width="300"/>
</div>

This University Project aims to provide a Framework which can be used to analyse a commercial and identify whether or not this commercial contains a Brand Differentiating Message (BDM). [This Spreadsheet](BDM.xlsx) contains a list of Superbowl Ads which are labeled with either 1 (BDM) or 0 (No BDM) by the marketing faculty of my University. Due to limitations in Data Quality (Subjectivity, Binary Classification) and Quantity (only a few hundred ads, class imbalance), this project should be considered as more of a conceptual approach, rather than a model with a high accuracy.

## CRISP-DM

This Project is structured using the CRISP-DM Framework

### 1 Business Understanding
### 2 Data Understanding
[Jupyter Notebook](<./app/2 - Data Understanding.ipynb>)
### 3 Data Preparation, 4 Modeling, 5 Evaluation
[Jupyter Notebook](<./app/3-5 Data Preparation, Modeling, Evaluation.ipynb>)
### 6 Deployment
[Webapp](<./app/6 Deployment.py>)

Due to the small number of Ads, we used a machine learning approach, with manually engineered features, rather than just passing text through to a neural network.

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

**Video Tutorial**
[![Watch Tutorial](https://www.flaticon.com/free-icon/video_4404094#)]
(https://github.com/NilsLeo/Commercial-Brand-Differentiating-Message-Analysis/raw/refs/heads/main/video_tutorial.mp4)



### Prerequisites
- Docker
- docker-compose
- An NVIDIA GPU. This image was tested on an RTX 4070. If you don't have access to one, you will likely need to make some changes to the [Dockerfile](./Dockerfile).
- Nvidia Container Toolkit. Follow these instructions: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#next-steps


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

#### Other Editors
If you prefer to use a different editor, you can work on the files in the [app](./app/) directory outside of the container. You will need to install the dependencies yourself on your host OS. These can be seen in the [Dockerfile](./Dockerfile) and the [requirements.txt](./app/requirements.txt)

#### Tips
If you want to persist the changes, make sure to update the requirements.txt when it comes to python packages and add any external dependencies to the [Dockerfile](./Dockerfile).

## Contributing
If you would like to contribute to this project, please feel free to fork the repository, create a feature branch, and then submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

[^1]: [Nvidia Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)
