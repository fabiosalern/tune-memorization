# An Approach for Extracting Training Data from fine-tuned Large Language Models for Code
Replication package for the Mac thesis titled: "*An Approach for Extracting Training Data from fine-tuned Large Language Models for Code*"

## Requirements
Ensure you have the following software installed on your machine:

- Python 3.8

The requirements can be installed running:
`pip install -r requirements.txt`

The code was intended to run on an Nvidia A100 with 80GB Vram, 32GB of RAM and 16 CPU cores. The extraction experiments can run on a single GPU. The fine-tuning requires multiple GPUs. Specifically, for StarCoder2-3B, 7B, and 15B, we employ two, four, and six GPUs.

## Data
In the `tune-memorization/data` directory you will find two subdirectories:
1. `tune-memorization/data/ftune-dataset`: This directory contains the script that automatically downloads and processes the data used for fine-tuning and evaluation.
2. `tune-memorization/data/samples`: This directory consists of samples to perform the code attacks and the code to replicate the sample generations:
    - `/pre-train`: This includes the dataset with the samples used to perform the *pre-train code attack*. To run the code and replicate the dataset construction, you need to have the Java subset of the-stack-v2-dedup loaded in your Huggingface cache folder. You can find it [HERE](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup/viewer/Java).

    - `/fine-tune`:  This includes the datasets with the samples used to perform the *fine-tune code attack* on different duplication rates (d=1, d=2, d=3, d>3). To run the code and replicate the dataset construction, you need to have the fine-tuning set loaded in your Huggingface cache folder.

## Training
In the `tune-memorization/training` directory you will find the fine-tuning scripts for each model size of the StarCoder2: `/scoder3b`, `/scoder7b`, `/scoder15b `. Additionally in the folder `/train-stats` we share figures of the fine-tuning stats. 

## Evaluation
In the `tune-memorization/evaluation` directory you will find three subdirectories:
1. `tune-memorization/evaluation/forgetting` The directory includes the following:
    - Data: Contains the data obtained from the pre-training code attack, corresponding to Research Questions 1 and 2 (RQ1, RQ2).
    - Evaluation Scripts: Scripts used to run the pre-training code attacks, organized by experiments.
    - Plots and Tables: Includes notebooks with all the plots and tables used in the associated paper, as well as additional ones not included in the publication.
2. `tune-memorization/evaluation/memorization` The directory includes the following:
    - Data: Contains the data obtained from the fine-tuning code attack, corresponding to Research Questions 3 (RQ3).
    - Evaluation Scripts: Scripts used to run the fine-tuning code attack, organized by experiments.
    - Plots and Tables: Includes notebooks with all the plots and tables used in the associated paper, as well as additional ones not included in the publication.
3. `tune-memorization/evaluation/data-inspection` The directory includes the following:
    - Data: containts the data obtained from the categorization task (RQ4)
    - Plots and Tables: Includes notebooks with all the plots and tables used in the associated paper, as well as additional ones not included in the publication.

## Ethical use
Please use the code and concepts shared here responsibly and ethically. The authors have provided this code to enhance the security and safety of large language models (LLMs). Avoid using this code for any malicious purposes. When disclosing data leakage, take care not to compromise individuals' privacy unnecessarily.

