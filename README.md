# MLOps as a Solution for Machine Learning Models

Creating an End-to-End MLOps pipelines using Azure ML and Azure Pipelines to help manage the Machine Learning model lifecycle.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project is about developing an End-to-End MLOPS pipelines developed from scratch that will allow us to extract and transform data, set up an environment, train and deploy machine learning models.

## Technologies
Project is created with:
* Azure Pipelines
* Azure Machine Learning
* Python 3.7
* YAML

## Setup
To run this project: 

* Clone the repository in your Azure Devops Organizations
* Go to your Azure DevOps Project
* Select Pipelines from the left hand blade
* Click the button for “New Pipeline”
* Select the “Azure Repos Git” option
* Select your repository
* On the configure tab, select “Existing Azure Pipelines YAML File”
* Select a pipeline '.yml' file as the path to your yaml file and click “Continue”
* After reviewing, click “Run”
* Repeat for each pipeline.
