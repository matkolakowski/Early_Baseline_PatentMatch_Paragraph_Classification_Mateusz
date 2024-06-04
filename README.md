# Early baseline for claim&cited-paragraph classification on PatentMatch

The corresponding Plan of Experiment is provided [here](https://www.notion.so/Early-baseline-for-claim-cited-paragraph-classification-on-PatentMatch-Mateusz-3bb38949ec454d21a3dfd91036ef7bf4).
\
The [Full Report](https://www.notion.so/Report-Mateusz-Early-Baseline-PatentMatch-Paragraph-Classification-2024-06-03-225701fd36884bbdaeae99efe7a4ca82) describing the effort is also available.

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

If you use Colab just run all the cells from the notebooks because the environment in Colab has to be created after each initialization.

### Set up MLFlow credentials

The code is designed to log model parameters using the MLFlow library. To use MLFlow seamlessly, create an .env file that will contain your MLFlow credentials.

If you haven't used MLFlow before, check out the tutorial at:
[MLFlow get started tutorial](https://mlflow.org/docs/latest/getting-started/index.html)

Rename example.env to .env. Keeping the .env name is necessary for the code to work properly.
`cp example.env .env` 

Open .env in any text editor or run command:
`nano .env`

Enter your authentication details and save the file.

MLFLOW_TRACKING_URI= your_mlflow_server_url
MLFLOW_TRACKING_USERNAME= your_user_name
MLFLOW_TRACKING_PASSWORD= your_password

Remember, never share your credentials in a public repository. This file is ignored in .gitignore already. 


## Data

A sample of the PatentMatch data used in this repository [is available at](https://drive.google.com/drive/folders/1zuTdW9Ke2hOC5vCLWgKywAVjqg_ZLYIJ?usp=drive_link) 

Copy test.parquet and train.parquet to data folder and run notebooks/data_preprocessing.ipynb to obtain processed data for the model.

You can provide your own training and test set. It is required that the sets be in the form of a list containing tuples
[(str(text), str(text_b), int(label)), ...] and were saved in the .json format.

It is not recommended to store data sets in a repository, so the data directory is included in .gitignore


## Run training

If you have already downloaded and processed the data and completed your MLFlow credentials, you can repeat this experiment.

Open notebooks/train_model.ipynb, edit the config variable, which contains the basic hyperparameters of the model.
Initialize LLMManager with the name of the model and tokenizer (default 'beta-base_uncased', but the code should work unchanged with models from the RoBERT family).
Run the training and if the results are satisfactory, save the model using the model.save_pretrained(path_to_model) method.
