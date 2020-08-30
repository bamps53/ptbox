import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import tqdm
from matplotlib import pyplot as plt

def dict_to_json(dict_obj, file_name):
    with open(file_name, 'w') as fp:
        json.dump(dict_obj, fp)

def find_best_threshold(preds, targs, do_plot=False):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

    accuracy_list = []
    f1_list = []
    recall_list=[]
    precision_list=[]
    results={}

    thrs = np.arange(0.1, 0.9, 0.01)
    for thr in tqdm.tqdm(thrs):
        accuracy_list.append(accuracy_score(targs, preds>thr))
        f1_list.append(f1_score(targs, preds>thr, average='samples'))
        recall_list.append(recall_score(targs, preds>thr, average='samples'))
        precision_list.append(precision_score(targs, preds>thr, average='samples'))

    result_df = pd.DataFrame({
        'accuracy':accuracy_list,
        'fi_score':f1_list,
        'recall':recall_list,
        'precision':precision_list,
        },
        index=thrs)


    accuracy = np.array(accuracy_list)
    f1 = np.array(f1_list)
    recall=np.array(recall_list)
    precision=np.array(precision_list)

    pm = accuracy.argmax()

    best_thr = thrs[pm]
    best_f1 = f1[pm].item()
    best_accuracy=accuracy[pm].item()
    best_recall=recall[pm].item()
    best_precision=precision[pm].item()

    results['best_thr']=best_thr
    results['best_accuracy']=best_accuracy
    results['best_f1']=best_f1
    results['recall']=best_recall
    results['precision']=best_precision

    for k,v in results.items():
        print(k, ':', v)

    if do_plot:
        plt.plot(thrs, f1)
        plt.vlines(x=best_thr, ymin=0, ymax=1)
        plt.text(best_thr+0.03, best_f1-0.01, f'$F_{2}=${best_f1:.3f}', fontsize=14);
        plt.show()

    return result_df


def calc_metrics(preds, targs, classwise):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

    preds = preds.argmax(axis=1)
    targs = targs.argmax(axis=1)

    results = {}

    results['accuracy'] = accuracy_score(targs, preds)
    results['macro_f1'] = f1_score(targs, preds, average='macro')
    #results['micro_f1'] = f1_score(targs, preds, average='micro')
    results['macro_recall'] = recall_score(targs, preds, average='macro')
    #results['micro_recall'] = recall_score(targs, preds, average='micro')
    results['macro_precision'] = precision_score(targs, preds, average='macro')
    #results['micro_precision'] = precision_score(targs, preds, average='micro')
    if classwise:
        classwise_acccuracy = classwise_acccuracy_score(preds, targs, return_counts=False)
        for i in range(len(classwise_acccuracy)):
            results['accuracy_class{}'.format(i)] = classwise_acccuracy[i]

    return results

def classwise_acccuracy_score(preds, targets, return_counts=False):
    all_accuracy = []
    class_count = []
    num_classes = len(np.unique(targets))
    for i in range(num_classes):
        class_idx = targets == i
        acc = (preds[class_idx] == targets[class_idx]).mean()
        all_accuracy.append(acc)
        class_count.append(class_idx.sum())
    if return_counts:
        return all_accuracy, class_count
    else:
        return all_accuracy