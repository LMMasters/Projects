import functools as ft
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sklearn.metrics as sklF
import csv

# g(f(x)) -> F([x, f, g, ...])
F = lambda z: [*ft.reduce(lambda x, y: map(y, x), [z[:1]] + z[1:])][0]

# Output
def output(data, model, record = True, data_cla = 2):
    y_pred = model.pred(data[0])
    y_pred = np.array(y_pred)
    y_pred_prob = model.pred_prob(data[0])
    auc = sklF.roc_auc_score(data[1], y_pred_prob)
    print("AUC: %0.4f" % auc)
    #if record == True:
        #hfw = h5py.File(record + '_' + str(auc) + ".h5", 'w')
        #hfw.create_dataset("output", data = y_pred_prob, dtype = "float32")
        #fout = open(record + ".auc", 'a')
        #fout.write("AUC: %0.4f\n" % auc)
        
        #with open('C:/Users/lucbu/Documents/Master Thesis/Results/scores/FCN.csv', 'w') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(y_pred)
            
    np.savetxt('C:/Users/lucbu/Documents/Master Thesis/Results/scores/FCN.csv', y_pred_prob)
    print("Accuracy: %.2f %%" %
            ((data[1] == y_pred).sum() / data[1].size * 100))
    wa, tot = [0] * data_cla, [0] * data_cla
    for i in range(data[1].size):
        if data[1][i] != y_pred[i]:
            wa[data[1][i]] += 1
        tot[data[1][i]] += 1
    print("Wrong Answer:", wa)
    print("Total:", tot)
    print("WA/Tot:", ["%.2f%%" % (wa[i] / tot[i] * 100)
                        if tot[i] else "N/A" for i in range(data_cla)])

def sig_cla(data):
    tot = {}
    for i in data:
        if not i in tot:
            tot[i] = 1
        else:
            tot[i] += 1
    return tot

def output_cla(data, model):
    tot1, tot2, wa = sig_cla(data[0]), {}, {}
    for pre, sig, bsm in zip(model.pred(data[1]), data[2], data[3]):
        if not bsm in tot2:
            wa[bsm] = 0
            tot2[bsm] = 1
        else:
            tot2[bsm] += 1
        if pre != sig:
            wa[bsm] += 1
    print(len(tot1))
    for i in sorted(tot1):
        print("%s, Total Train: %d" % (i, tot1[i]))
    print(len(tot2))
    for i in sorted(tot2):
        print("%s, Wrong Answer: %d, Total Test: %d" % (i, wa[i], tot2[i]))
    print(len(wa))
    for i in sorted(wa):
        print("%s, Accuracy: %.2f%%"
                % (i, (1 - wa[i] / tot2[i]) * 100 if tot2[i] else "N/A"))

def draw_roc(data, model, name, record = None):
    y_pred_prob = model.pred_prob(data[0])
    auc = sklF.roc_auc_score(data[1], y_pred_prob)
    print(name + ": AUC = %0.4f" % auc)
    fpr, tpr, thresholds = sklF.roc_curve(data[1], y_pred_prob)
    if record:
        hfw = h5py.File(record + '_' + str(auc) + ".h5", 'w')
        hfw.create_dataset("fpr", data = fpr, dtype = "float32")
        hfw.create_dataset("tpr", data = tpr, dtype = "float32")
    plt.figure()
    plt.title(name + ": AUC = %0.4f" % auc)
    plt.plot(fpr, tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.savefig(name + ".pdf")
    plt.close()

def draw_cla(data, model = None, method = None,
                name = "hist", data_cla = 2, hist_config = None):
    assert data_cla < 7
    plt.figure()
    cla = [[] for i in range(data_cla)]
    for i in range(data[1].size):
        cla[data[1][i]].append(data[0][i])
    if model:
        predict = [np.array, model.pred_prob]
        for i in range(data_cla):
            cla[i] = F([cla[i]] + predict)
        if method[-4:] == "lgbm":
            plt.title("LightGBM Classifier")
        if method[-3:] == "xgb":
            plt.title("XGBoost Classifier")
        if method[-2:] == "nn":
            plt.title("NN Classifier")
    else:
        plt.title(name)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    if not hist_config:
        hist_config = {"density": True, "bins": 100, "histtype": "step"}
    for i in range(data_cla):
        plt.hist(cla[i], color = colors[i], **hist_config)
    plt.ylabel("Density")
    plt.xlabel("Value");
    plt.savefig(name + "Hist.pdf")
    plt.close()

def write_XY_h5(data, name):
    hfw = h5py.File(name, 'w')
    hfw.create_dataset("X1", data = data[0], dtype = "float32")
    hfw.create_dataset("Y1", data = data[1], dtype = "int8")
    hfw.create_dataset("X2", data = data[2], dtype = "float32")
    hfw.create_dataset("Y2", data = data[3], dtype = "int8")
    hfw.create_dataset('Z', data = data[4], dtype = "int8")
    hfw.close()

def data_info(name, *data):
    print(name)
    for i in data:
        print(i.shape)
        print(i.dtype)