import pickle
import argparse
import dataAnalysis as da
import training as ta
import shap_analysis as sa
import pandas as pd


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='training bilstm model for text classification')
    parser.add_argument('-train_path',default = "data/train.csv",type=str,help='training data path')
    parser.add_argument('-load_model',default = None,type=str,help='path of already trained model')
    parser.add_argument('-tokenizer',default = None,type=str,help='path of already calculated word index')
    parser.add_argument('-huaodata',default = None,type=str,help='path of human annotated text')

    args = parser.parse_args()
    train_padded, train_label, word_index,comment_text_ha, label_ha = da.main(args.train_path, False, args.tokenizer, args.huaodata)
    if args.load_model is None:
        model, x_train, x_val, y_train, y_val = ta.main(train_padded, train_label)
    else:
        model, x_train, x_val, y_train, y_val = ta.main(train_padded, train_label,True, args.load_model)

    filtered_response = []
    category_label = {i:x for i,x in enumerate(list(y_val.columns))}
    for x, y in zip(x_val, y_val.values):
        count= 0
        for yy in y:
            if yy == 1:
                count += 1
        if count ==6:
            filtered_response.append(x)

    sa.main(model,x_train, x_val, word_index, filtered_response, category_label,comment_text_ha, label_ha)
