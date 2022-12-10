import os
import functools as ft
import xgboost as xgb
import lightgbm as lgbm
import joblib
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch_geometric as geo
import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from src.yuFunctional import npf2i
import numpy as np

# g(f(x)) -> F(x, f, g...)
# g(f([x1, x2...])) -> FF([x1, x2...], f, g...)
def F(*z):
    z = list(z)
    return [*ft.reduce(lambda x, y: map(y, x), [z[:1]] + z[1:])][0]
FF = lambda *z: [*ft.reduce(lambda x, y: map(y, x), z)]
# f(x1, x2..., y1, y2...) -> fyx(f, y1, y2...)(x1, x2...)
fyx = lambda f, *x: lambda *y: f(*y, *x)

# XGBoost
class xgb_model(object):
    def __init__(self, x_train = None, x_test = None,
                    y_train = None, y_test = None,
                    xgb_para = None, fit_para = None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.xgb_para = xgb_para
        self.fit_para = fit_para
    def train(self):
        self.model = xgb.XGBClassifier(**self.xgb_para)
        start = time.time()
        self.model.fit(
            self.x_train, self.y_train,
            eval_set = [(self.x_train, self.y_train),
                        (self.x_test, self.y_test)], **self.fit_para
        )
        print("Time: %.0fs" % (time.time() - start))
        return self
    def save(self, name):
        joblib.dump(self.model, name)
        return self
    def load(self, name):
        self.model = joblib.load(name)
        return self
    def pred(self, data):
        return self.model.predict(data)
    def pred_prob(self, data):
        return self.model.predict_proba(data)[:, 1]

# LightGBM
class lgbm_model(object):
    def __init__(self, x_train = None, x_test = None,
                    y_train = None, y_test = None,
                    lgbm_para = None, fit_para = None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.lgbm_para = lgbm_para
        self.fit_para = fit_para
    def train(self):
        self.model = lgbm.LGBMClassifier(**self.lgbm_para)
        start = time.time()
        self.model.fit(
            self.x_train, self.y_train,
            eval_set = [(self.x_train, self.y_train),
                        (self.x_test, self.y_test)], **self.fit_para
        )
        if self.lgbm_para["objective"] == "binary":
            plt.figure()
            lgbm.plot_metric(self.model, metric = "binary_logloss")
            plt.savefig("lgbm_loss.pdf")
            plt.figure()
            lgbm.plot_metric(self.model, metric = self.fit_para["eval_metric"])
            plt.savefig("lgbm_metric.pdf")
        print("Time: %.0fs" % (time.time() - start))
        return self
    def save(self, name):
        joblib.dump(self.model, name)
        return self
    def load(self, name):
        self.model = joblib.load(name)
        return self
    def pred(self, data):
        return self.model.predict(data)
    def pred_prob(self, data):
        return self.model.predict_proba(data)[:, 1]

#class lgbm_llh_model(object):
#    def __init__(self, x_train = None, x_test = None,
#                    y_train = None, y_test = None,
#                    llh_train = None, llh_test = None,
#                    lgbm_para = None, fit_para = None):
#        super().__init__(x_train, x_test, y_train, y_test, lgbm_para, fit_para)
#        self.llh_train = llh_train
#        self.llh_test = llh_test
#    def train(self):

# NN
class nn_model(object):
    np_vec = lambda x: x.float().detach().cpu().numpy().flatten()
    def __init__(self, X_train = None, X_test = None,
                    Y_train = None, Y_test = None,
                    layers = None, nn_para = None):
        if nn_para != None:
            self.X_train = torch.from_numpy(X_train)
            self.X_test = torch.from_numpy(X_test)
            self.Y_train = torch.from_numpy(Y_train)
            self.Y_test = torch.from_numpy(Y_test)
            self.layers = layers
            self.bs = nn_para["batch_size"]
            self.lr = nn_para["learning_rate"]
            self.learn_fn = nn_para["learn_fn"]
            self.optimizer = nn_para["optimizer"]
            self.metric = nn_para["metric"]
            self.epochs = nn_para["epochs"]
            self.device = nn_para["device"]
            self.network = nn_para["network"]
            self.estop = nn_para["early_stopping"]
            self.scheduler = nn_para["scheduler"]
            if (nn_para["metric_better"] == '>'):
                self.better = lambda x, y: x > y
            if (nn_para["metric_better"] == '<'):
                self.better = lambda x, y: x < y
    def train(self):
        start = time.time()
        train_set = Data.TensorDataset(self.X_train, self.Y_train)
        val_set = Data.TensorDataset(self.X_test, self.Y_test)
        train_loader = Data.DataLoader(
            train_set,
            sampler = Data.RandomSampler(train_set),
            batch_size = self.bs,
            drop_last = True
        )
        val_loader = Data.DataLoader(
            val_set,
            sampler = Data.SequentialSampler(val_set),
            batch_size = self.bs,
            drop_last = False
        )
        self.network = [nn.BatchNorm1d(self.X_train.shape[1]),
                        nn.Linear(self.X_train.shape[1],
                                    self.layers[0])] + self.network
        self.model = nn.Sequential(*self.network).to(self.device)
        optim = self.optimizer(self.model.parameters(), lr = self.lr)
        if self.scheduler:
            scheduler = self.scheduler(optim)
        checkpoint = []
        train_losses, train_metrics, val_losses, val_metrics = [], [], [], []
        for epoch in range(self.epochs):
            print("======= Epoch %s / %s =======" % (epoch + 1, self.epochs))
            tot_loss = 0
            metric_pred = []
            metric_true = []
            self.model.train()
            for x_train, y_train in tqdm(train_loader):
                optim.zero_grad()
                y_pred = self.model(x_train.to(self.device)).float()
                y_train = y_train.view(-1, 1).float().to(self.device)
                loss = self.learn_fn(y_pred, y_train, reduction = "sum")
                tot_loss += loss
                metric_pred.append(y_pred)
                metric_true.append(y_train)
                loss.backward()
                optim.step()
            if self.scheduler:
                scheduler.step()
            list_to_np = [torch.cat, nn_model.np_vec]
            train_metric = self.metric(
                F(metric_pred, *list_to_np),
                F(metric_true, *list_to_np)
            )
            print("Train Loss: %.5f, \t\tTrain Metric: %.7f"
                    % (tot_loss / len(train_loader), train_metric))
            train_losses.append(tot_loss / len(train_loader))
            train_metrics.append(train_metric)
            print("Running Validation...")
            self.model.eval()
            tot_loss = 0
            metric_pred = []
            metric_true = []
            with torch.no_grad():
                for x_test, y_test in tqdm(val_loader):
                    y_pred = self.model(x_test.to(self.device)).float()
                    y_test = y_test.view(-1, 1).float().to(self.device)
                    loss = self.learn_fn(y_pred, y_test, reduction = "sum")
                    tot_loss += loss
                    metric_pred.append(y_pred)
                    metric_true.append(y_test)
                val_metric = self.metric(
                    F(metric_pred, *list_to_np),
                    F(metric_true, *list_to_np)
                )
                print("Validation Loss: %.5f, \tValidation Metric: %.7f"
                        % (tot_loss / len(val_loader), val_metric))
                val_losses.append(tot_loss / len(val_loader))
                val_metrics.append(val_metric)
                # Early Stopping
                if self.estop:
                    if len(checkpoint)\
                        and self.better(checkpoint[0], val_metric):
                        if len(checkpoint) == self.estop\
                            or epoch + 1 == self.epochs:
                            print("Early Stopping at Epoch: %s"
                                    % (epoch - len(checkpoint) + 1))
                            self.model = torch.load("best_nn.model")
                            break
                        checkpoint.append(val_metric)
                    elif epoch + 1 != self.epochs:
                        checkpoint = [val_metric]
                        torch.save(self.model, "best_nn.model")
        self.model.eval().cpu()
        if os.path.isfile("best_nn.model"):
            os.remove("best_nn.model")
        plt.figure()
        plt.plot(train_losses, "r-", val_losses, "g-")
        plt.xlabel("epochs", fontsize = 16)
        plt.ylabel("loss", fontsize = 16)
        plt.title("Metric during training", fontsize = 16)
        plt.legend(
            labels = ["train", "validation"],
            loc = "upper right"
        )
        plt.savefig("nn_loss.pdf")
        plt.figure()
        plt.plot(train_metrics, "r-", val_metrics, "g-")
        plt.xlabel("epochs", fontsize = 16)
        plt.ylabel("metric", fontsize = 16)
        plt.title("Metric during training", fontsize = 16)
        plt.legend(
            labels = ["train", "validation"],
            loc = "upper right"
        )
        plt.savefig("nn_metric.pdf")
        print("Time: %.1fs" % (time.time() - start))
        return self
    def save(self, name):
        torch.save(self.model, name)
        return self
    def load(self, name):
        self.model = torch.load(name).cpu().eval()
        return self
    def pred(self, data):
        return F(data, torch.from_numpy, self.model, nn_model.np_vec, npf2i)
    def pred_prob(self, data):
        return F(data, torch.from_numpy, self.model, nn_model.np_vec)

################################################################################


class ParticleStaticEdgeConv(geo.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = "max")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels[0], bias = False),
            geo.nn.BatchNorm(out_channels[0]), 
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias = False),
            geo.nn.BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias = False),
            geo.nn.BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )
    def forward(self, x, edge_index, k):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x = x)
    def message(self, edge_index, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)
        out_mlp = self.mlp(tmp)
        return out_mlp
    def update(self, aggr_out):
        return aggr_out

class ParticleDynamicEdgeConv(ParticleStaticEdgeConv):
    def __init__(self, in_channels, out_channels, k = 7):
        super().__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias = False),
            geo.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()
    def forward(self, pts, fts, batch = None):
        edges = geo.nn.knn_graph(
            pts, self.k, batch, loop = False, flow = self.flow
        )
        aggrg = super().forward(fts, edges, self.k)
        x = self.skip_mlp(fts)
        out = torch.add(aggrg, x)
        return self.act(out)

class ParticleNet(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()
        previous_output_shape = settings["input_features"]
        self.input_bn = geo.nn.BatchNorm(settings["input_features"])
        self.conv_process = torch.nn.ModuleList()
        for layer_param in settings["conv_params"]:
            K, channels = layer_param
            self.conv_process.append(
                ParticleDynamicEdgeConv(previous_output_shape, channels, k = K)
            )
            previous_output_shape = channels[-1]
        self.fc_process = torch.nn.ModuleList()
        for layer_param in settings["fc_params"]:
            drop_rate, units = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p = drop_rate),
                torch.nn.ReLU()
            )
            self.fc_process.append(seq)
            previous_output_shape = units
        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, 1)
        self.output_activation = torch.nn.Sigmoid()
    def forward(self, batch):
        fts = self.input_bn(batch.x)
        pts = batch.pos
        for layer in self.conv_process:
          fts = layer(pts, fts, batch.batch)
          pts = fts
        x = geo.nn.global_mean_pool(fts, batch.batch)
        for layer in self.fc_process:
            x = layer(x)
        x = self.output_mlp_linear(x)
        x = self.output_activation(x)
        return x

class particleNet_model(object):
    np_vec = lambda x: x.float().detach().cpu().numpy().flatten()
    def __init__(self, X_train = None, X_test = None,
                    Y_train = None, Y_test = None, pn_para = None):
        if pn_para != None:
            self.X_train = torch.from_numpy(X_train)
            self.X_test = torch.from_numpy(X_test)
            self.Y_train = torch.from_numpy(Y_train)
            self.Y_test = torch.from_numpy(Y_test)
            self.device = pn_para["device"]
            self.bs = pn_para["batch_size"]
            self.lr = pn_para["learning_rate"]
            self.learn_fn = pn_para["learn_fn"]
            self.optimizer = pn_para["optimizer"]
            self.metric = pn_para["metric"]
            self.epochs = pn_para["epochs"]
            self.devie = pn_para["device"]
            self.estop = pn_para["early_stopping"]
            self.scheduler = pn_para["scheduler"]
            if (pn_para["metric_better"] == '>'):
                self.better = lambda x, y: x > y
            if (pn_para["metric_better"] == '<'):
                self.better = lambda x, y: x < y
            self.settings = pn_para["pn_settings"]
            self.settings["input_features"] = 3
        else:
            self.bs = 512
            self.device = "cpu" if not torch.cuda.is_available() else "cuda"
    @staticmethod
    def genGraphs(data):
        graphs = []
        for event, label in zip(data[0], data[1]):
            fts_temp = [torch.tensor([-1, event[6], 0])]
#           pts_temp = [torch.tensor([0, 0, event[7]])]
            pts_temp = [torch.tensor([0, event[7]])]
            max_obj = 4, 4, 1, 1, 1, 1
            pos_obj = 8, 24, 40, 44, 48, 52
#           obj = -1, -2, 1, 1, 2, 2
            t2i = lambda x: int(x.item())
            for i in range(6):
                for j in range(t2i(event[i])):
                    if j < max_obj[i]:
                        fts_temp.append(
                            torch.tensor([i, event[pos_obj[i] + 4 * j],
                                            event[pos_obj[i] + 4 * j + 1]])
                        )
                        pts_temp.append(
#                           torch.tensor([obj[i], event[pos_obj[i] + 4 * j + 2],
                            torch.tensor([event[pos_obj[i] + 4 * j + 2],
                                            event[pos_obj[i] + 4 * j + 3]])
                        )
                    else:
                        fts_temp.append(torch.tensor([i, 0, 0]))
#                       pts_temp.append(torch.tensor([obj[i], 0, 0]))
                        pts_temp.append(torch.tensor([0, 0]))
            fts_temp = torch.stack(fts_temp)
            pts_temp = torch.stack(pts_temp)
            graph = geo.data.Data(
                x = fts_temp,
                pos = pts_temp,
                y = label
            )
            graphs.append(graph)
        return graphs
    def train(self):
        start = time.time()
        print(self.device)
        train_graphs = particleNet_model.genGraphs((self.X_train, self.Y_train))
        val_graphs = particleNet_model.genGraphs((self.X_test, self.Y_test))
        train_loader = geo.loader.DataLoader(
            train_graphs,
            batch_size = self.bs,
            shuffle = True,
            drop_last = True
        )
        val_loader = geo.loader.DataLoader(
            val_graphs,
            batch_size = self.bs,
            shuffle = True,
            drop_last = False
        )
        self.model = ParticleNet(self.settings).to(self.device)
        optim = self.optimizer(self.model.parameters(), lr = self.lr)
        if self.scheduler:
            scheduler = self.scheduler(optim)
        checkpoint = []
        train_losses, train_metrics, val_losses, val_metrics = [], [], [], []
        for epoch in range(self.epochs):
            print("======= Epoch %s / %s =======" % (epoch + 1, self.epochs))
            tot_loss = 0
            metric_pred = []
            metric_true = []
            self.model.train()
            for batch in tqdm(train_loader):
                optim.zero_grad()
                y_pred = self.model(batch.to(self.device)).float()
                y_train = batch.y.view(-1, 1).float().to(self.device)
                loss = self.learn_fn(y_pred, y_train, reduction = "sum")
                tot_loss += loss
                metric_pred.append(y_pred)
                metric_true.append(y_train)
                loss.backward()
                optim.step()
            if self.scheduler:
                scheduler.step()
            list_to_np = [torch.cat, particleNet_model.np_vec]
            train_metric = self.metric(
                F(metric_pred, *list_to_np),
                F(metric_true, *list_to_np)
            )
            print("Train Loss: %.5f, \t\tTrain Metric: %.7f"
                    % (tot_loss / len(train_loader), train_metric))
            train_losses.append(tot_loss / len(train_loader))
            train_metrics.append(train_metric)
            print("Running Validation...")
            self.model.eval()
            tot_loss = 0
            metric_pred = []
            metric_true = []
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    y_pred = self.model(batch.to(self.device)).float()
                    y_test = batch.y.view(-1, 1).float().to(self.device)
                    loss = self.learn_fn(y_pred, y_test, reduction = "sum")
                    tot_loss += loss
                    metric_pred.append(y_pred)
                    metric_true.append(y_test)
                val_metric = self.metric(
                    F(metric_pred, *list_to_np),
                    F(metric_true, *list_to_np)
                )
                print("Validation Loss: %.5f, \tValidation Metric: %.7f"
                        % (tot_loss / len(val_loader), val_metric))
                val_losses.append(tot_loss / len(val_loader))
                val_metrics.append(val_metric)
                # Early Stopping
                if self.estop:
                    if len(checkpoint)\
                        and self.better(checkpoint[0], val_metric):
                        if len(checkpoint) == self.estop\
                            or epoch + 1 == self.epochs:
                            print("Early Stopping at Epoch: %s"
                                    % (epoch - len(checkpoint) + 1))
                            self.model = torch.load("best_pn.model")
                            break
                        checkpoint.append(val_metric)
                    elif epoch + 1 != self.epochs:
                        checkpoint = [val_metric]
                        torch.save(self.model, "best_pn.model")
        self.model.eval().cpu()
        if os.path.isfile("best_pn.model"):
            os.remove("best_pn.model")
        plt.figure()
        plt.plot(train_losses, "r-", val_losses, "g-")
        plt.xlabel("epochs", fontsize = 16)
        plt.ylabel("loss", fontsize = 16)
        plt.title("Metric during training", fontsize = 16)
        plt.legend(
            labels = ["train", "validation"],
            loc = "upper right"
        )
        plt.savefig("PN_loss.pdf")
        plt.figure()
        plt.plot(train_metrics, "r-", val_metrics, "g-")
        plt.xlabel("epochs", fontsize = 16)
        plt.ylabel("metric", fontsize = 16)
        plt.title("Metric during training", fontsize = 16)
        plt.legend(
            labels = ["train", "validation"],
            loc = "upper right"
        )
        plt.savefig("PN_metric.pdf")
        print("Time: %.0fs" % (time.time() - start))
        return self
    def save(self, name):
        torch.save(self.model, name)
        return self
    def load(self, name):
        self.model = torch.load(name).cpu().eval()
        return self
    def pred(self, data):
        return F(data, self.pred_prob, npf2i)
    def pred_prob(self, data):
        model = self.model.to(self.device)
        graphs = particleNet_model.genGraphs((data, torch.zeros(data.shape[0])))
        loader = geo.loader.DataLoader(
            graphs,
            batch_size = self.bs,
            shuffle = False,
            drop_last = False
        )
        pred = []
        for graph in tqdm(loader):
            pred.append(
                F(graph.to(self.device), model, particleNet_model.np_vec)
            )
        return np.concatenate(pred)
