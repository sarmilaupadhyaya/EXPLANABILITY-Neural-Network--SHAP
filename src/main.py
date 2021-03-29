import argparse
import dataAnalysis as da
import training as ta


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='training bilstm model for text classification')
    parser.add_argument('-train_path',default = "data/train.csv",type=str,help='training data path')
    parser.add_argument('-load_model',default = None,type=str,help='path of already trained model')
    args = parser.parse_args()
    train_padded, train_label, word_index = da.main(args.train_path)
    if args.load_model is None:
        model, x_train, x_val, y_train, y_val = ta.main(train_padded, train_label)
    else:
        model, x_train, x_val, y_train, y_val = ta.main(train_padded, train_label,True, args.load_model)
    import pdb
    pdb.set_trace()
