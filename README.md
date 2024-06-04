# Early baseline for claim&cited-paragraph classification on PatentMatch

The corresponding <a href:=https://www.notion.so/Plan-of-Experiment-PoE-template-efed4153dd7849c5979e9abb00293ec0>Plan of Experiment is provided here</a>.
\
The <a href:=https://www.notion.so/Report-Mateusz-Early-Baseline-PatentMatch-Paragraph-Classification-2024-06-03-225701fd36884bbdaeae99efe7a4ca82>Full Report</a> describing the effort is also available.

## Goal of the experiment
1. On-hands discovery of the PatentMatch dataset, obtaining very preliminary baseline quality on the claim&cited-paragraph classification task (2 texts on model input, one binary classification label on output).
2. Learn how to use the newly introduced standards for implementation of research experiments.

## Get started
The code in this repository requires a GPU to provide reasonable computation time. If you don't have a cuda-compatible GPU on your local machine, you can use external providers such as Colab.
All code was written on a Colab virtual machine with a T4 GPU and is Colab-friendly.
I strongly recommend that if you don't have a GPU or the skills to set it up, start working with this repo on Colab.

### System parameters of Colab env:

DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_DESCRIPTION="Ubuntu 22.04.3 LTS"

Python 3.10.12

Colab runtime environment: T4 GPU

### How to download repo

`!git clone git@github.com:matkolakowski/Early_Baseline_PatentMatch_Paragraph_Classification_Mateusz.git`

### Set up environment

If you use local machine, create new environment and install dependencies from requirements.txt

`pip install -r requirements.txt`

### Set up MLFlow credentials

The code is designed to log model parameters using the MLFlow library. To use MLFlow seamlessly, create an .env file that will contain your MLFlow credentials.

If you haven't used MLFlow before, check out the tutorial at:
[MLFlow get started tutorial] (https://mlflow.org/docs/latest/getting-started/index.html)



## Data
Do provide a description of how to obtain the necessary data. Specifically, do not store data files larger than 1 MB in the data folder.

## Credentials
Remember to provide all credentials, keys in the .env file only. This file is ignored in .gitignore already.  
The `example.env` file is a placeholder to demonstrate what credential keys are needed and should be pushed into the repository.  
