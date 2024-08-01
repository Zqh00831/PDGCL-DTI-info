from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import conv
from utils import *
import numpy as np
import torch as t
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class ContrastiveLossModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ContrastiveLossModule, self).__init__()
        self.contrastive_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        return self.contrastive_layer(x)

class Model(nn.Module):
    def __init__(self, sizes, pro_sim, drug_sim):
        super(Model, self).__init__()
        np.random.seed(sizes.seed)
        t.manual_seed(sizes.seed)
        self.pro_size = sizes.pro_size
        self.drug_size = sizes.drug_size
        self.F1 = sizes.F1
        self.F2 = sizes.F2
        self.F3 = sizes.F3

        self.seed = sizes.seed
        self.h1_gamma = sizes.h1_gamma
        self.h2_gamma = sizes.h2_gamma
        self.h3_gamma = sizes.h3_gamma

        self.lambda1 = sizes.lambda1
        self.lambda2 = sizes.lambda2

        self.kernel_len = 4

        self.att_d = Parameter(t.ones((1, 4)), requires_grad=True)
        self.att_m = Parameter(t.ones((1, 4)), requires_grad=True)

        self.pro_sim = t.DoubleTensor(pro_sim)
        self.drug_sim = t.DoubleTensor(drug_sim)

        self.gcn_1 = conv.GATConv(self.pro_size + self.drug_size, self.F1)
        self.gcn_2 = conv.GATConv(self.F1, self.F2)
        self.gcn_3 = conv.GATConv(self.F2, self.F3)

        self.alpha2 = t.randn(self.drug_size, self.pro_size).double()
        self.alpha1 = t.randn(self.pro_size, self.drug_size).double()

        self.weight_matrix = t.randn(32, 32)
        self.pro_l = []
        self.drug_l = []

        self.pro_k = []
        self.drug_k = []

        self.contrastive_module1 = ContrastiveLossModule(input_dim=self.drug_size, hidden_dim=self.drug_size)
        self.contrastive_module2 = ContrastiveLossModule(input_dim=self.pro_size, hidden_dim=self.pro_size)

        self.fc1_1 = nn.Linear(self.drug_size, 256)
        self.fc1_2 = nn.Linear(self.pro_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        t.manual_seed(self.seed)
        x = input['feature']
        adj = input['Adj']
        pro_kernels = []
        drug_kernels = []

        H1 = t.relu(self.gcn_1(x, adj['edge_index']))
        pro_kernels.append(t.DoubleTensor(getGipKernel(H1[:self.pro_size].clone(), 0, self.h1_gamma, True).double()))
        drug_kernels.append(t.DoubleTensor(getGipKernel(H1[self.pro_size:].clone(), 0, self.h1_gamma, True).double()))

        H2 = t.relu(self.gcn_2(H1, adj['edge_index']))
        pro_kernels.append(t.DoubleTensor(getGipKernel(H2[:self.pro_size].clone(), 0, self.h2_gamma, True).double()))
        drug_kernels.append(t.DoubleTensor(getGipKernel(H2[self.pro_size:].clone(), 0, self.h2_gamma, True).double()))

        H3 = t.relu(self.gcn_3(H2, adj['edge_index']))
        pro_kernels.append(t.DoubleTensor(getGipKernel(H3[:self.pro_size].clone(), 0, self.h3_gamma, True).double()))
        drug_kernels.append(t.DoubleTensor(getGipKernel(H3[self.pro_size:].clone(), 0, self.h3_gamma, True).double()))

        pro_kernels.append(self.pro_sim)
        drug_kernels.append(self.drug_sim)

        pro_k = sum([self.att_m[0][i] * pro_kernels[i] for i in range(self.kernel_len)])
        self.pro_k = normalized_kernel(pro_k)
        drug_k = sum([self.att_d[0][i] * drug_kernels[i] for i in range(self.kernel_len)])
        self.drug_k = normalized_kernel(drug_k)
        self.pro_l = laplacian(pro_k)
        self.drug_l = laplacian(drug_k)

        out1 = self.pro_l.mm(self.alpha1)
        out2 = self.drug_l.mm(self.alpha2)

        X1 = out1.detach().cpu().numpy()
        y1 = np.random.randint(2, size=X1.shape[0])       
        ada1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)
        ada1.fit(X1, y1)
        boosted_features1 = t.tensor(ada1.predict(X1), dtype=t.float32).to(out1.device).unsqueeze(1)

        X2 = out2.detach().cpu().numpy()
        y2 = np.random.randint(2, size=X2.shape[0])  
        ada2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)
        ada2.fit(X2, y2)
        boosted_features2 = t.tensor(ada2.predict(X2), dtype=t.float32).to(out2.device).unsqueeze(1)

        contrastive_out1 = self.contrastive_module1(boosted_features1)
        contrastive_out2 = self.contrastive_module2(boosted_features2)

        feature1 = self.dropout(contrastive_out1)
        feature1 = t.tensor(feature1, dtype=t.float32)
        output1 = self.fc1_1(feature1)
        output1 = self.relu(output1)
        output1 = self.dropout(output1)

        output1 = self.fc2(output1)
        output1 = self.relu(output1)
        output1 = self.dropout(output1)

        output1 = self.fc3(output1)
        output1 = self.relu(output1)
        output1 = self.dropout(output1)

        output1 = self.fc4(output1)
        output1 = self.sigmoid(output1)

        feature2 = self.dropout(contrastive_out2)
        feature2 = t.tensor(feature2, dtype=t.float32)
        output2 = self.fc1_2(feature2)
        output2 = self.relu(output2)
        output2 = self.dropout(output2)

        output2 = self.fc2(output2)
        output2 = self.relu(output2)
        output2 = self.dropout(output2)

        output2 = self.fc3(output2)
        output2 = self.relu(output2)
        output2 = self.dropout(output2)

        output2 = self.fc4(output2)
        output2 = self.sigmoid(output2)

        output = t.mm(t.mm(output1, self.weight_matrix), output2.T)
        return output