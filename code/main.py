import numpy as np
import scipy.sparse as sp
import random
import gc
import os
import scipy.io as scio
import time
import csv
from clac_metric import get_metrics
from utils import constructHNet, constructNet, get_edge_index, load_data, Sizes
import torch as t
from torch import optim
from loss import Myloss
import DTI_CLGAT
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def train(model, train_data, optimizer, sizes):
    model.train()
    regression_crit = Myloss()

    def train_epoch():
        model.zero_grad()
        score = model(train_data)
        tloss = regression_crit(train_data['Y_train'], score, model.pro_l, model.drug_l, model.alpha1, model.alpha2,
                                sizes)
        tloss = tloss.requires_grad_()
        model.alpha1 = t.mm(
            t.mm((t.mm(model.pro_k, model.pro_k) + model.lambda1 * model.pro_l).inverse(), model.pro_k),
            2 * train_data['Y_train'] - t.mm(model.alpha1, model.drug_k.T)).detach()
        model.alpha2 = t.mm(
            t.mm((t.mm(model.drug_k, model.drug_k) + model.lambda2 * model.drug_l).inverse(), model.drug_k),
            2 * train_data['Y_train'].T - t.mm(model.alpha2, model.pro_k.T)).detach()

        tloss.backward(retain_graph=True)
        optimizer.step()
        return tloss

    for epoch in range(1, sizes.epoch + 1):
        train_reg_tloss = train_epoch()
        print("epoch : %d, loss:%.2f" % (epoch, train_reg_tloss.item()))
    pass


def PredictScore(train_pro_drug_matrix, pro_matrix, drug_matrix, seed, sizes):
    np.random.seed(seed)
    train_data = {}
    train_data['Y_train'] = t.DoubleTensor(train_pro_drug_matrix)
    feature = constructHNet(train_pro_drug_matrix, pro_matrix, drug_matrix)
    feature = t.FloatTensor(feature)
    train_data['feature'] = feature
    adj = constructNet(train_pro_drug_matrix)
    adj = t.FloatTensor(adj)
    adj_edge_index = get_edge_index(adj)
    train_data['Adj'] = {'data': adj, 'edge_index': adj_edge_index}
    model = DTI_CLGAT.Model(sizes, pro_matrix, drug_matrix)
    print(model)
    for parameters in model.parameters():
        print(parameters, ':', parameters.size())
    optimizer = optim.Adam(model.parameters(), lr=sizes.learn_rate)
    train(model, train_data, optimizer, sizes)
    return model(train_data)


def random_index(index_matrix, sizes):
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)
    random.shuffle(random_index)
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + random_index[association_nam - association_nam % k_folds:]
    return temp


def crossval_index(drug_pro_matrix, sizes):
    random.seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(drug_pro_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_pro_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = [pos_index[i] + neg_index[i] for i in range(sizes.k_fold)]
    return index


def cross_validation_experiment(drug_pro_matrix, drug_matrix, pro_matrix, sizes):
    index = crossval_index(drug_pro_matrix, sizes)
    metric = np.zeros((1, 7))
    pre_matrix = np.zeros(drug_pro_matrix.shape)
    print("seed=%d, evaluating drug-microbe...." % (sizes.seed))
    metric_p = []
    all_true_labels = []
    all_pred_proba = []

    for k in range(sizes.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        split_start_time_stamp = time.time()
        train_matrix = np.array(drug_pro_matrix, copy=True)
        train_matrix[tuple(np.array(index[k]).T)] = 0
        drug_len = drug_pro_matrix.shape[0]
        pro_len = drug_pro_matrix.shape[1]
        drug_pro_res = PredictScore(train_matrix, drug_matrix, pro_matrix, sizes.seed, sizes)
        predict_y_proba = drug_pro_res.reshape(drug_len, pro_len).detach().numpy()
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]
        metric_tmp = get_metrics(drug_pro_matrix[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)])

        # Collect true labels and predicted probabilities for ROC and AUPR
        all_true_labels.extend(drug_pro_matrix[tuple(np.array(index[k]).T)].flatten())
        all_pred_proba.extend(predict_y_proba[tuple(np.array(index[k]).T)].flatten())

        print(f"{k + 1} folds time cost: {time.time() - split_start_time_stamp}")
        folder = os.path.join('results', f"{sizes.seed}", f"{data_set}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_name = f"fold{k + 1}-{metric_tmp[1]}-{metric_tmp[0]}-predict"
        file_path = os.path.join(folder, f"{file_name}.mat")
        scio.savemat(file_path, {'label': drug_pro_matrix[tuple(np.array(index[k]).T)],
                                 'predict': predict_y_proba[tuple(np.array(index[k]).T)]})
        result_name = 'result.csv'
        file_path1 = os.path.join(folder, result_name)
        print(metric_tmp)
        metric += metric_tmp
        metric_p.append(metric_tmp)
        metric_t = np.vstack((metric, metric_p)).T
        np.savetxt(file_path1, metric_t, delimiter=",")
        del train_matrix
        gc.collect()

    # Save true labels and predicted probabilities
    with open('true_labels_and_pred_proba.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True Label', 'Predicted Probability'])
        for true_label, pred_proba in zip(all_true_labels, all_pred_proba):
            writer.writerow([true_label, pred_proba])

    # Calculate and plot ROC curve
    fpr, tpr, _ = roc_curve(all_true_labels, all_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC_Curve.png')
    plt.show()

    # Calculate and plot AUPR curve
    precision, recall, _ = precision_recall_curve(all_true_labels, all_pred_proba)
    average_precision = average_precision_score(all_true_labels, all_pred_proba)
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label='AUPR curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig('AUPR_Curve.png')
    plt.show()

    print(metric / sizes.k_fold)
    metric = np.array(metric / sizes.k_fold)
    final_file_name = f"final-auroc_{metric[0][1]}-aupr_{metric[0][0]}-predict"
    file_path2 = os.path.join(folder, f"{final_file_name}.mat")
    scio.savemat(file_path2, {'label': drug_pro_matrix, 'predict': pre_matrix})
    return metric, pre_matrix


if __name__ == "__main__":
    data_path = '../data/dataset/'
    data_set = 'Drugbank/'

    drug_sim = np.loadtxt(data_path + data_set + 'drugsimilarity.txt', delimiter='\t')
    pro_sim = np.loadtxt(data_path + data_set + 'proteinsimilarity.txt', delimiter='\t')
    adj_triple = np.loadtxt(data_path + data_set + 'adj.txt')
    drug_pro_matrix = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),
                                    shape=(len(drug_sim), len(pro_sim))).toarray()
    # labels = t.Tensor(drug_pro_matrix)
    csv_file_path = 'drug_pro_matrix.csv'

    # 将 drug_pro_matrix 转换为列表格式
    drug_pro_matrix_list = drug_pro_matrix.tolist()
    # 打开 CSV 文件并写入数据
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(drug_pro_matrix_list)  # 写入数据行
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    sizes = Sizes(drug_sim.shape[0], pro_sim.shape[0])
    results = []

    result, pre_matrix = cross_validation_experiment(drug_pro_matrix, drug_sim, pro_sim, sizes)
    print(list(sizes.__dict__.values()) + result.tolist()[0][:2])
