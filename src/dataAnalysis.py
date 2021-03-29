import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def read_csv(csv_path):

    ### Reading csv data files using pandas dataframe

    train = pd.read_csv(csv_path, encoding = "ISO-8859-1")

    return train




def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)      ### conversion of contraction words to expanded words
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)                                                 ### removing non-word characters
    text = re.sub('[^A-Za-z\' ]+', '',text)                                        ### removing all non-alphanumeric values(Except single quotes)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    text = ' '.join([word for word in text.split() if word not in (stop_words)])    ### Stopwords removal
    return text



def data_analysis(train):

    categorywise_data = train.drop(['id', 'comment_text'], axis=1)
    counts_category = []                       
    categories = list(categorywise_data.columns.values)
    for i in categories:
       counts_category.append((i, categorywise_data[i].sum()))

    dataframe = pd.DataFrame(counts_category, columns=['Labels', 'number_of_comments'])   

    dataframe.plot(x='Labels', y='number_of_comments', kind='bar',figsize=(8,8))
    plt.title("Number of comments per category")
    plt.ylabel('No. of Occurrences', fontsize=12)
    plt.xlabel('Labels', fontsize=12)

    dataframe = pd.DataFrame(pd.DataFrame(train[train.columns[2:]].sum(axis=1)).reset_index()[0].value_counts())
    dataframe["Total no. of sentences"]=dataframe[0]
    dataframe["Total No. of labels in a sentence"]=dataframe.index
    dataframe.plot(x="Total No. of labels in a sentence", y="Total no. of sentences", kind='bar',figsize=(8,8))
    plt.title("No of comments based on the count of labels")
    plt.ylabel('Total no. of sentences', fontsize=12)
    plt.xlabel('Total No. of labels in a sentence', fontsize=12)

    fig, plots = plt.subplots(2,3,figsize=(15,12))
    plot1, plot2, plot3, plot4, plot5, plot6 = plots.flatten()
    sns.countplot(train['obscene'], ax = plot1)
    sns.countplot(train['threat'], ax = plot2)
    sns.countplot(train['insult'], ax = plot3)
    sns.countplot(train['identity_hate'], ax = plot4)
    sns.countplot(train['toxic'], ax = plot5)
    sns.countplot(train['severe_toxic'], ax = plot6)

def main(train_path, analysis=False):
    
    train = read_csv(train_path)

    train_data = train["comment_text"]
    train_label=train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]
    

    if analysis:
        data_analysis(train)

    tokenizer = Tokenizer(num_words = 40000) #40000 words are used here
    tokenizer.fit_on_texts(train_data)

    #convert each text into array of integers with help of tokenizer.
    train_final = tokenizer.texts_to_sequences(train_data)
    train_padded =pad_sequences(train_final, maxlen=150)


    return train_padded, train_label, tokenizer.word_index



#main("data/train.csv","data/test.csv", True)





