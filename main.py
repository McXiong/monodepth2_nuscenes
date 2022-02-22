import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam

from model.disp_net import DispNet
from NUSC import Nusc_ds
from mmcv import Config
from loss_fn import LossFn
import time

def get_path_prefix():
    machine = os.uname()[1]
    prefix = ''
    # Local machine
    if machine == '253':
        prefix = '/home/user/Documents/gpu5'
    return prefix


def save_model(save_path, current_model, current_epoch, marker, timestamp):
    save_path = os.path.join(save_path, timestamp)
    save_to = os.path.join(save_path, '{}_{}.pkl'.format(marker, current_epoch))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(current_model, save_to)
    print('<== Model is saved to {}'.format(save_to))


def print_nn(mm):
    def count_pars(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    num_pars = count_pars(mm)
    print(mm)
    print('# pars: {}'.format(num_pars))
    print('{} : {}'.format('device', device))


def create_ds(data_root):
    nusc = Nusc_ds()
    # TODO: finish this
    train_ds, val_ds = nusc
    return train_ds, val_ds


def create_tr_dl(dataset):
    dl_tr = DataLoader(dataset, batch_size=4, shuffle=True)
    return dl_tr


def create_val_dl(dataset):
    val_dl = DataLoader(dataset, batch_size=4, shuffle=False)
    return val_dl


def training_setting(model, opts):
    optimizer = Adam(model.parameters(), lr=0.0001)
    sch = MultiStepLR(optimizer , milestones=[15], gamma=0.5)
    # TODO: finish this
    loss_fn = LossFn(opts)
    return optimizer, loss_fn, sch


def train(dl, optimizer, loss_fn, epoch):
    losses = 0.
    counter = 1
    tmp_losses = 0.
    tmp_counter = 0

    model.eval()
    for idx, data in enumerate(dl):
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)

        loss = loss_fn(out, data)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        counter += 1
        tmp_losses += loss.item()
        tmp_counter += 1

        if idx % 5 == 0:
            print('  [{}][{}] loss: {:.4f}'.format(epoch, idx, tmp_losses / tmp_counter))

            tmp_losses = 0.
            tmp_counter = 0

    print('##> [{}] Train loss: {:.4f}'.format(epoch, losses / counter))


def eval(dl, loss_fn):
    losses = 0.
    counter = 0
    model.eval()
    with torch.no_grad():
        for idx, (data, v_la, a_la) in enumerate(dl):
            data = data.to(device)
            v_la = v_la.to(device)
            a_la = a_la.to(device)

            out = model(data)
            del data
            loss_v = loss_fn(out, v_la)
            loss_a = loss_fn(out, a_la)
            loss = loss_v + loss_a * 0

            losses += loss.item()
            counter += 1
    print('==> Val loss: {:.4f}'.format(losses / counter))


if __name__ == "__main__":
    machine = get_path_prefix()
    data_root = machine + ''
    save_path = machine + ''
    save_every = 1
    val_every = 1
    epoches = 100
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    file = "../configs/mono2.yaml"
    cfg = Config.fromfile(file)

    train_ds, val_ds = create_ds(data_root)
    train_dl = create_tr_dl(train_ds)
    val_dl = create_val_dl(val_ds)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DispNet(cfg.model)
    model.to(device)
    optimizer, loss_fn, sch = training_setting(model)

    print_nn(model)

    for epoch in range(1, epoches + 1):
        train(train_dl, optimizer, loss_fn, epoch)
        if epoch % save_every == 0:
            save_model(save_path, model, epoch, "v_a", timestamp)

        if epoch % val_every == 0:
            eval(val_dl, loss_fn)
