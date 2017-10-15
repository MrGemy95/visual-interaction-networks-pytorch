import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.pool = nn.MaxPool2d(2, 2)

        # visual encoder modules
        self.conv1 = nn.Conv2d(10, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        # shared linear layer to convert the tensors to shape N_obj*64
        self.fc1 = nn.Linear(128, 3 * 64)
        # shared MLP layer to output the encoded state code N_obj*64
        self.fc2 = nn.Linear(64 * 2, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        # end of visual encoder

        # dynamic predictor modules
        self.self_cores = {}
        for i in range(3):
            self.self_cores[i] = []
            self.self_cores[i].append(nn.Linear(64, 64))
            self.self_cores[i].append(nn.Linear(64, 64))
        self.rel_cores = {}
        for i in range(3):
            self.rel_cores[i] = []
            self.rel_cores[i].append(nn.Linear(64 * 2, 64))
            self.rel_cores[i].append(nn.Linear(64, 64))
            self.rel_cores[i].append(nn.Linear(64, 64))

        self.affector = {}
        for i in range(3):
            self.affector[i] = []
            self.affector[i].append(nn.Linear(64, 64))
            self.affector[i].append(nn.Linear(64, 64))
            self.affector[i].append(nn.Linear(64, 64))
        self.out = {}
        for i in range(3):
            self.out[i] = []
            self.out[i].append(nn.Linear(64 + 64, 64))
            self.out[i].append(nn.Linear(64, 64))
        self.aggregator1 = nn.Linear(64 * 3, 64)
        self.aggregator2 = nn.Linear(64, 64)

    def core(self, s, core_idx):
        objects = torch.chunk(s, 3, 1)

        s_reshaped = s.view(-1, 64)
        self_sd_h1 = F.relu(self.self_cores[core_idx][0](s_reshaped))
        self_sd_h2 = self.self_cores[core_idx][1](self_sd_h1) + self_sd_h1
        self_dynamic = self_sd_h2.view(-1, 3, 64)

        rel_combination = []
        for i in range(6):
            row_idx = int(i / (2));
            col_idx = int(i % (2));
            rel_combination.append(torch.cat([objects[row_idx], objects[col_idx]], 1))
        rel_combination = torch.cat(rel_combination)

        rel_sd_h1 = F.relu(self.rel_cores[core_idx][0](rel_combination))
        rel_sd_h2 = F.relu(self.rel_cores[core_idx][1](rel_sd_h1) + rel_sd_h1)
        rel_sd_h3 = self.self_cores[core_idx][2](rel_sd_h2) + rel_sd_h2
        rel_objects = torch.chunk(rel_sd_h3, 6)
        obj1 = rel_objects[0] + rel_objects[1]
        obj2 = rel_objects[2] + rel_objects[3]
        obj3 = rel_objects[4] + rel_objects[5]
        rel_dynamic = torch.stack([obj1, obj2, obj3], 1)
        dynamic_pred = self_dynamic + rel_dynamic
        dynamic_pred = dynamic_pred.view(-1, 64)
        aff1 = F.relu(self.affector[core_idx][0](dynamic_pred))
        aff2 = F.relu(self.affector[core_idx][1](aff1) + aff1)
        aff3 = self.affector[core_idx][2](aff2) + rel_sd_h2
        aff3 = aff3.view(-1, 3, 64)
        aff_s = torch.cat([aff3, s], 2)
        aff_s = aff_s.view(-1, 64 + 64)

        out1 = F.relu(self.out[core_idx][0](aff_s))
        out2 = self.out[core_idx][1](out1) + out1
        out2 = out2.view(-1, 3, 64)

        return out2

    def forward(self, x, x_cor, y_cor):
        f1, f2, f3, f4, f5, f6 = torch.chunk(x, 6)
        f1f2 = torch.cat([f1, f2], 1)
        f2f3 = torch.cat([f2, f3], 1)
        f3f4 = torch.cat([f3, f4], 1)
        f4f5 = torch.cat([f4, f5], 1)
        f5f6 = torch.cat([f5, f6], 1)

        pairs = torch.cat([f1f2, f2f3, f3f4, f4f5, f5f6])
        pairs = torch.cat([pairs, x_cor, y_cor], dim=1)
        ve_h1 = F.relu(self.conv1(pairs))
        ve_h1 = self.pool(ve_h1)
        ve_h2 = F.relu(self.conv2(ve_h1) + ve_h1)
        ve_h2 = self.pool(ve_h2)
        ve_h3 = F.relu(self.conv3(ve_h2) + ve_h2)
        ve_h3 = self.pool(ve_h3)
        ve_h4 = F.relu(self.conv4(ve_h3) + ve_h3)
        ve_h4 = self.pool(ve_h4)
        ve_h5 = F.relu(self.conv5(ve_h4) + ve_h4)
        ve_h5 = self.pool(ve_h5)
        unit_pairs = ve_h5.view(-1, 128)
        # p1,p2,p3,p4,p5=torch.chunk(unit_pairs, 5)
        encoded_pairs = self.fc1(unit_pairs)
        p1, p2, p3, p4, p5 = torch.chunk(encoded_pairs, 5)
        p1 = p1.view(-1, 3, 64)
        p2 = p2.view(-1, 3, 64)
        p3 = p3.view(-1, 3, 64)
        p4 = p4.view(-1, 3, 64)
        p5 = p5.view(-1, 3, 64)

        pair1 = torch.cat([p1, p2], 2)
        pair2 = torch.cat([p2, p3], 2)
        pair3 = torch.cat([p3, p4], 2)
        pair4 = torch.cat([p4, p5], 2)

        diff_pairs = torch.cat([pair1, pair2, pair3, pair4])
        diff_pairs = diff_pairs.view(-1, 64 * 2)
        shared_h1 = F.relu(self.fc2(diff_pairs))
        shared_h2 = F.relu(self.fc2(shared_h1) + shared_h1)
        shared_h3 = self.fc2(diff_pairs) + shared_h2
        state_codes = shared_h3.view(-1, 3, 64)
        s1, s2, s3, s4 = torch.chunk(state_codes, 4)
        rolls = []
        for i in range(20):
            c1 = self.core(s1, 0)
            c2 = self.core(s2, 1)
            c3 = self.core(s3, 2)
            all_c = torch.cat([c1, c2, c3], 2)
            all_c = all_c.view(-1, 64 * 3)
            aggregator1 = F.relu(self.aggregator1(all_c))
            aggregator2 = self.aggregator2(aggregator1)
            aggregator2 = aggregator2.view(-1, 3, 64)
            rolls.append(aggregator2)
            s1, s2, s3, s4 = s2, s3, s4, aggregator2
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
