# Explaining BILSTM Model for text classification using SHAP value estimation

This project is an implementation of BILSTM model for toxicity classification. The dataset were published in kaggle. (Link)[]
The implementation is explained using the SHAP method. Following is the structure of the project.


## Table of Contents

- [Introduction](#Introduction): information about this project
- [Setup](#Setup): How to set up the project
- [Project Structure](#Project-Structure): File structure of project
- - [Data Preprocessing](#Data-Preprocessing): POS extraction, aggregation
- - [Models](#Models): training
- - [Explanation Method](#Explanation-Method): Shap implementation


## Introduction

This is the project submitted to Explanability for Neural Network seminar for 2020 winter semester.

# Setup

- using pip

```
virtualenv <envname>
pip install -r requirements.txt
```

- using conda

```
conda env update --file requirements.txt
```

# Project-Structure
```
├── data  # data
├
├── src├ # Source code for preprocessing, tokenization
├
├      ├──shapAnalysis.py #explanation method shap
├      ├──main.py   #code to run everything from model training, explanation to visualization
├      ├──dataAnalysis.py    #splitting data, tokenizing and padding
├      ├──embedding.py     #extracts and save embedding for sentence       
├      ├── training.py  #training and testing code                         
├      ├
├── requirements.txt  #requirements
├
└── README.md         #read me

```

1. RUN ALL
First create virtual Environment and activate it.

This code will read data, train model, analyse the data and finally explain it and save the figures inside data folder
```
python src/main.py -huaodata data/human_test_S.csv


```

2.  RUN ALL with data analysis

```
python src/main.py -huaodata data/human_test_S.csv -analysis True


```

3. If you want to load the model and tokenizer already trained, provide the path of them. The default path for model and tokenizer os inside data.

```
python3 src/main.py -huaodata data/human_test_S.csv -load_model data/<model name> -tokenizer data/<tokenizer name>

```


