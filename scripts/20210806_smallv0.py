import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--logdir", required=True)
args = parser.parse_args()


print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


data_path = args.data
MODEL_DIR = args.logdir


class LargeFileDataset(torch.utils.data.IterableDataset):
  def __init__(self, dirpath: str, start=None, end=None) -> None:
    self.files = glob.glob("{}/*".format(dirpath))
    self.start = 0 if start is None else start
    self.end = len(self.files) if end is None else end
    self.fid = self.start
    self.lid = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self.fid == self.end:
      raise StopIteration
    if self.lid == 0:
      with open(self.files[self.fid]) as f:
        self.lines = f.readlines()

    ret = self.lines[self.lid]
    feature, action = self.lines[self.lid].strip().split('\t')
    x = torch.FloatTensor(list(map(float,feature.split(' '))))
    y = int(action)

    self.lid += 1
    if self.lid == len(self.lines):
        self.lid = 0
        self.fid += 1

    return x, y

def eval_acc(net, test_dataloader):
    total = {'all': 0}
    correct = {'all': 0}
    def action_type(code):
        if code < 37:
            return 'discard'
        if code < 74:
            return 'tsumogiri'
        if code < 101:
            return 'chi'
        if code < 141:
            return 'pon'
        if code < 175:
            return 'kan'
        if code < 176:
            return 'tsumo'
        if code < 177:
            return 'ron'
        if code < 178:
            return 'riichi'
        if code < 179:
            return 'kyusyu'
        if code < 180:
            return 'no'
        raise('Invalid action code')
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        outputs = net(x)
        _, pred = torch.max(outputs.data, 1)

        total['all'] += int(y.size(0))
        correct['all'] += int((pred == y).sum().item())

        for p, t in zip(pred, y):
            action = action_type(t)
            if action in total:
                total[action] += 1
                if p == t:
                    correct[action] += 1
            else:
                total[action] = 1
                if p == t:
                    correct[action] = 1
                else:
                    correct[action] = 0

    return total, correct

def save(net, name):
    model_path = MODEL_DIR + "/" + name + ".pt"
    model_on_cpu = net.cpu()
    torch.save(model_on_cpu.state_dict(), model_path)
    traced_script_module = torch.jit.script(model_on_cpu)
    torch.jit.save(traced_script_module, model_path)

class Net(torch.nn.Module):
    def __init__(self, num_layers=0, num_units=100):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units

        # current_hand: 4*34
        # target_tile: 37 
        # under_riichi: 1 
        # river: 4*34*4
        # open_tiles: 4*34*4
        # who: 4
        # shanten: 1
        self.linears = torch.nn.ModuleList([torch.nn.Linear(
            10*34, num_units)])
        self.linears.extend([torch.nn.Linear(num_units, num_units) for i in range(num_layers)])
        # action: 0~179
        self.fc = torch.nn.Linear(num_units, 180)

    def forward(self, x):
        for linear in self.linears:
          x = linear(x)
          x = F.relu(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(num_units: int, num_layers: int, lr: float, batch_size: int) -> float:
    net = Net(num_layers, num_units).to(device)
    print(net)
    loss_fn = torch.nn.NLLLoss()
    trial_name = f"small_v0-lr={lr}-num_layers={num_layers}-num_units={num_units}"
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    cnt = 0
    START=0
    TEST_NUM = 6000
    TOTAL_NUM = 60000
    EPOCH_NUM = 1
    for epoch in range(EPOCH_NUM):
        print('epoch:', epoch)
        train_dataset = LargeFileDataset(data_path, start=TEST_NUM + START, end=TOTAL_NUM + START)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = net(x)
            loss = loss_fn(outputs, y)
            _, pred = torch.max(outputs.data, 1)
            acc = (pred == y).sum().item() / y.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cnt % 1000 == 0:
                test_dataset = LargeFileDataset(data_path, start=0 + START, end=TEST_NUM + START)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
                #test_acc = eval_acc(net, test_dataloader)
                #print(f"test_acc: {test_acc}")
                test_total, test_correct = eval_acc(net, test_dataloader)
                print("test_acc: {}".format(test_correct['all'] / test_total['all']))
                for key, val in test_total.items():
                    if (key == 'all'): continue
                    print("  {}: {}({} / {})".format(key, test_correct[key] / val, test_correct[key], val))
            # if cnt % 100 == 0:
            #     print(f"{cnt: 08d} (epoch {epoch}) loss:{loss.item(): .04f} acc:{acc: .04f}")

            cnt += 1

    test_dataset = LargeFileDataset(data_path, start=0 + START, end=TEST_NUM + START)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    #test_acc = eval_acc(net, test_dataloader)
    #print(f"test_acc: {test_acc}")
    test_total, test_correct = eval_acc(net, test_dataloader)
    print("test_acc: {}".format(test_correct['all'] / test_total['all']))
    for key, val in test_total.items():
        if (key == 'all'): continue
        print("  {}: {}({} / {})".format(key, test_correct[key] / val, test_correct[key], val))
    save(net, trial_name)

train(2048, 2, 1e-3, 256) 
