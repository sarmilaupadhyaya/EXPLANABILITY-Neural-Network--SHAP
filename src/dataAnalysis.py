import os
import pickle
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

def save_tokenizer(wordindex):

    with open("data/wordindex","wb") as f:
        pickle.dump(wordindex, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("word index saved")

def load_tokenizer(word_index):

    with open(word_index, "rb") as f:
            tokenizer  = pickle.load(f)

    return tokenizer



def get_data_all_label(train):

    df = train[(train.toxic == 1) & (train.severe_toxic==1) & (train.obscene==1) & (train.threat==1) & (train.insult==1)&(train.identity_hate==1)]
    print("Comments that has all labels:")
    print("\n".join(df["comment_text"].tolist()))

def data_analysis(train):

    get_data_all_label(train)
    categorywise_data = train.drop(['id', 'comment_text'], axis=1)
    counts_category = []                       
    categories = list(categorywise_data.columns.values)
    for i in categories:
       counts_category.append((i, categorywise_data[i].sum()))

    dataframe = pd.DataFrame(counts_category, columns=['Labels', 'number_of_comments'])   

    dataframe.plot(x='Labels', y='number_of_comments', kind='bar',figsize=(10,8), color="red")
    plt.title("Number of comments per category", fontsize=20)
    plt.ylabel('No. of Occurrences', fontsize=20)
    plt.xlabel('Labels', fontsize=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.show()

    dataframe = pd.DataFrame(pd.DataFrame(train[train.columns[2:]].sum(axis=1)).reset_index()[0].value_counts())
    dataframe["Total no. of sentences"]=dataframe[0]
    dataframe["Total No. of labels in a sentence"]=dataframe.index
    dataframe.plot(x="Total No. of labels in a sentence", y="Total no. of sentences", kind='bar',figsize=(8,8))
    plt.title("No of comments based on the count of labels")
    plt.ylabel('Total no. of sentences', fontsize=12)
    plt.xlabel('Total No. of labels in a sentence', fontsize=12)
    plt.tight_layout()
    plt.show()

    fig, plots = plt.subplots(2,3,figsize=(15,12))
    plot1, plot2, plot3, plot4, plot5, plot6 = plots.flatten()
    sns.countplot(train['obscene'], ax = plot1)
    sns.countplot(train['threat'], ax = plot2)
    sns.countplot(train['insult'], ax = plot3)
    sns.countplot(train['identity_hate'], ax = plot4)
    sns.countplot(train['toxic'], ax = plot5)
    sns.countplot(train['severe_toxic'], ax = plot6)
    plt.show()
    target_data = train.drop(['id', 'comment_text'], axis=1)
    corrMatrix = target_data.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.show()



def main(train_path, analysis=False, word_index=None, huaodata=None):
    
    train = read_csv(train_path)
    if huaodata is not None:
        huaodata = read_csv(huaodata)
        

    train_data = train["comment_text"]
    train_label=train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]
    

    if analysis:
        data_analysis(train)

    tokenizer = Tokenizer(num_words = 40000) #40000 words are used here
    tokenizer.fit_on_texts(train_data)

    if word_index is None:
        save_tokenizer(tokenizer)
    else:
        tokenizer = load_tokenizer(word_index)

    train_final = tokenizer.texts_to_sequences(train_data)
    train_padded =pad_sequences(train_final, maxlen=150)

    if huaodata is not None:
        comment_text_ha = pad_sequences(tokenizer.texts_to_sequences(huaodata["comment_text"]), maxlen=150)
        label_ha = huaodata[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]

    return train_padded, train_label, tokenizer.word_index, comment_text_ha, label_ha
