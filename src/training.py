from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input
from sklearn.model_selection import train_test_split



def get_model():

    model = Sequential()
    model.add(Embedding(40000, 128))
    model.add(LSTM(units = 64, dropout = 0.2,return_sequences=True))
    model.add(LSTM(units = 64, dropout = 0.2))
    model.add(Dense(units = 6, activation = 'sigmoid'))

    return model

def train(model, x_train, x_val, y_train, y_val):

    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["AUC"])
    model.fit(x_train, y_train, batch_size = 32, epochs = 1, validation_data = (x_val, y_val))

    return model



def loading_model(model_path):
    
    return keras.models.load_model(model_path)


def main(train_padded=None, train_label=None,load_model=False, model_path="data/modelbilstm.h5"):
    
    x_train, x_val, y_train, y_val = train_test_split(train_padded, train_label, shuffle = True, random_state = 123)
    if not load_model:
        model = get_model()
        model = train(model, x_train, x_val, y_train, y_val)
        save_model(model, model_path)
    else:
        model = loading_model(model_path)
    return model, x_train,x_val,y_train, y_val
