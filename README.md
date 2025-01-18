# Analysis Framework for Identifying Brand Differentiating Messages (BDM) in Commercials
<center>
<div style="text-align: center;">
  <img src="./Resources/images/SuperBowl.png" alt="Super Bowl" width="300"/>
</div>
</center>

This University Project aims to provide a Framework which can be used to analyse a commercial and identify whether or not this commercial contains a Brand Differentiating Message (BDM). [This Spreadsheet](BDM.xlsx) contains a list of Superbowl Ads which are labeled with either 1 (BDM) or 0 (No BDM) by the marketing faculty of my University. Due to limitations in Data Quality (Subjectivity, Binary Classification) and Quantity (only a few hundred ads, class imbalance), this project should be considered as more of a conceptual approach, rather than a model with a high accuracy.

Due to the small number of Ads, we used a machine learning approach, with manually engineered features, rather than just passing text through to a nerual network.

## Prerequisites
- Docker
- docker-compose

## Installation
- simply run `docker-compose up -d`

## Usage
### Demo Application
open http://localhost:8502 to access the Demo Web UI. This will allow you to upload a Video and see the steps that are Model goes through before giving its prediction whether or not the Video contains a string BDM or not.
### Development
The source code is directly mapped to the [app](./app/) directory. This means that any changes made to the directory inside or outside of the container will be synced both ways.

#### Jupyter 

Open http://localhost:8889 to access the integrated Jupyter Development environment. The necessary dependencies are already installed.

In order to authenticate and set a password, in your terminal run `docker exec -it app sh` to enter the container and type in `jupyter server list` to retrieve the token

#### Other Editors

If you prefer to use a different editor, you can work on the files in the app [app](./app/) directory outside of the container. You will need to install the dependencies yourself on your host os. These can be seen in the [Dockerfile](./Dockerfile)

#### Tips

If you want to persist the changes, make sure to update the requirements.txt when it comes to python packages and add any external dependencies to the [Dockerfile](./Dockerfile)
