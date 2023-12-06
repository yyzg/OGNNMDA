# -*- coding:utf-8 -*-
# @Time:    2023/6/13 18:40
# @Author:  YYZG
#
from pathlib import Path
from typing import Union
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix,classification_report


def get_metrics(
        real_score: Union[np.ndarray, torch.Tensor],
        predict_score: Union[np.ndarray, torch.Tensor],
        draw: bool = False):
    if isinstance(real_score, torch.Tensor):
        real_score = real_score.cpu().numpy()
    if isinstance(predict_score, torch.Tensor):
        predict_score = predict_score.cpu().numpy()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    res = [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]
    if draw:
        return res, [fpr, tpr, precision_list, auc[0, 0], aupr[0, 0]]
    return res


# def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
#     test_index = np.where(train_matrix == 0)  # 原本所有的负样本(无边) + 验证集的边
#     real_score = interaction_matrix[test_index]
#     predict_score = predict_matrix[test_index]
#     return get_metrics(real_score, predict_score)


def get_metrics1(
        real_score: Union[np.ndarray, torch.Tensor],
        predict_score: Union[np.ndarray, torch.Tensor],
        draw: bool = False):
    if isinstance(real_score, torch.Tensor):
        real_score = real_score.cpu().numpy()
    if isinstance(predict_score, torch.Tensor):
        predict_score = predict_score.cpu().numpy()
    # fpr, tpr, thresholds = roc_curve(real_score, predict_score)
    precision,recall,thresholds = precision_recall_curve(real_score,predict_score)
    TP = np.zeros_like(thresholds, dtype='int')
    FP = np.zeros_like(thresholds, dtype='int')
    TN = np.zeros_like(thresholds, dtype='int')
    FN = np.zeros_like(thresholds, dtype='int')
    for i in range(thresholds.shape[0]):
        predict_res = np.where(predict_score > thresholds[i], 1, 0)
        TP[i] = np.sum(predict_res[real_score == 1] == real_score[real_score == 1])
        FP[i] = np.sum(predict_res) - TP[i]
        FN[i] = np.sum(real_score) - TP[i]
        TN[i] = real_score.shape[0] - TP[i] - FP[i] - FN[i]

    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    # precision, recall, thresholds = precision_recall_curve(real_score, predict_score)
    aupr_val = auc(recall, precision)
    auc_val = auc(fpr, tpr)
    f1_scores = 2 * precision * recall / (precision + recall)
    # best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])

    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    # best_f1_score_index = np.argmax(tpr)
    threshold = thresholds[best_f1_score_index]

    predict_score1 = np.zeros_like(predict_score)
    predict_score1[predict_score > threshold] = 1
    cm = confusion_matrix(real_score, predict_score1)
    print(classification_report(real_score,predict_score1))
    ((TN, FP), (FN, TP)) = cm
    print(f'TN FP FN TP\n: {cm}')
    accuracy_val = (TP + TN) / cm.sum()
    specificity = TN / (TN + FP)
    res = [aupr_val, auc_val, f1_scores[best_f1_score_index], accuracy_val, recall[best_f1_score_index], specificity,
           precision[best_f1_score_index]]
    if draw:
        # precision
        return res, [fpr, tpr, precision, auc_val, aupr_val]
    return res

