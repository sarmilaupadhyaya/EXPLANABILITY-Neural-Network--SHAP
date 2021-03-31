import shap
import matplotlib.pyplot as plt
import numpy as np

def save_single_class(explainer,shap_values,x_test_words, class_num, output_dir="data"):
    """
    """

    new_s = []
    new_x_test_words = []
    for x, y in zip(shap_values[0],x_test_words[0]):
        if y != "NONE":
            new_s.append(x)
            new_x_test_words.append(y)

    new_s = np.array(new_s)
    new_x_test_words_test_words = np.array(new_x_test_words)

    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.tight_layout()

    shap.force_plot(explainer.expected_value[0], new_s, new_x_test_words_test_words,matplotlib=True, show=False)
    plt.xlabel("Feature importance for class "+ str(class_num), fontsize=20)
    plt.savefig( "data/class"+str(class_num)+".pdf", bbox_inches='tight', format="pdf", dpi=600)
    plt.close()
    return new_s, new_x_test_words_test_words

def human_evaluation(explainer, comment_text_ha, label_ha, category_label, num2word):
    """

    """

    x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), comment_text_ha[i]))) for i in range(len(comment_text_ha))])
    values_labels = dict()
    total_label = dict()
    for tokens,comment, classes in zip(x_test_words, comment_text_ha, label_ha.values):
        shap_values = explainer.shap_values(np.array([comment]))
        for i, shap_value in enumerate(shap_values):
            cal_words = []
            label = category_label[i]
            if isinstance(classes[i], str):
                words = [x.strip() for x in classes[i].split(",")]
                words = [word for word in words if word not in [" ","",''," "]]
            else:
                words = []
            indices = sorted(range(len(shap_value[0])), key=lambda i: shap_value[0][i])[-10:]

            cal_words = [tokens[i] for i in indices]
            common = set(words).intersection(set(cal_words))
            
            if label not in values_labels:
                if len(words)>0:
                    values_labels[label] = len(common)/len(words)
                else:
                    values_labels[label] = 0
            else:
                if len(words)>0:
                    values_labels[label] += len(common)/len(words)
                else:
                    pass

            if label not in total_label:
                if len(words)>0:
                    total_label[label] = 1
            else:
                if len(words)>0:
                    total_label[label] += 1

                else:
                    pass

    values_labels = {key: value/total_label[key] for key, value in values_labels.items()}

    keys = values_labels.keys()
    values = values_labels.values()
    plt.bar(keys, values)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel("Different Classes", fontsize=20)
    plt.ylabel("Accuracy for Human Interpretation vs SHAP interpretation", fontsize=20)
    plt.show()
    plt.savefig("data/human_interpretation.pdf", format="pdf", dpi=600)





def main(model,x_train, x_val, wordIndex, filtered, category_label, comment_text_ha, label_ha):
    """
    """
    


    import numpy as np
    filtered = np.array(filtered)
    explainer = shap.DeepExplainer(model, x_train[:100])
    shap_values = explainer.shap_values(filtered[1:2])
    words = wordIndex
    num2word = {}
    for w in words.keys():
        num2word[words[w]] = w
    x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), filtered[i]))) for i in range(1,2)])
    final  = dict()
    for i in range(0, len(shap_values)):
        shap_value, word= save_single_class(explainer, shap_values[i], x_test_words,category_label[i])
        final[category_label[i]] = [shap_value, word]

    shap_values = explainer.shap_values(filtered)
    x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), filtered[i]))) for i in range(len(filtered))])
    shap.summary_plot(shap_values,x_test_words,feature_names=num2word,max_display=20)
    plt.show()
    plt.savefig( "data/class.pdf", bbox_inches='tight', format="pdf", dpi=600)
    plt.close()
    human_evaluation(explainer, comment_text_ha, label_ha, category_label, num2word)
    





