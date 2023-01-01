from importlib.metadata import metadata
import numpy as np
import pandas as pd
import torch as th
from torch.nn.functional import mse_loss
from hierarchy_data import (
    CustomHierarchyData,
    normalize_data,
    unnormalize_data,
)
from models.fnpmodels import EmbedMetaAttenSeq, RegressionSepFNP, Corem
from utils import lag_dataset_2
from models.utils import float_tensor, long_tensor
from tqdm import tqdm
import properscoring as ps
from random_split import random_split

# from torch.utils.tensorboard import SummaryWriter

SEED = 42
DEVICE = "cuda"
AHEAD = 8
REF_BACKUP_TIME = 1
BACKUP_TIME = 50
PRE_BATCH_SIZE = 10
PRE_TRAIN_LR = 0.001
PRE_TRAIN_EPOCHS = 1
C = 5.0
BATCH_SIZE = 10
TRAIN_LR = 0.001
LAMBDA = 0.1
TRAIN_EPOCHS = 1

np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)

if DEVICE == "cuda":
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
else:
    device = th.device("cpu")

data_obj = CustomHierarchyData()

# Let's create dataset
full_data = data_obj.data
train_data_raw = full_data
train_means = np.mean(train_data_raw, axis=1)
train_std = np.std(train_data_raw, axis=1)
train_data = (train_data_raw - train_means[:, None]) / train_std[:, None]
num_stocks = train_data.shape[0]

dataset_raw = lag_dataset_2(train_data, data_obj.shapes, REF_BACKUP_TIME, BACKUP_TIME)


class SeqDataset(th.utils.data.Dataset):
    def __init__(self, dataset):
        self.R, self.X, self.Y = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.R[idx], self.X[idx], self.Y[idx]


dataset = SeqDataset(dataset_raw)
train_dataset, val_dataset = random_split(dataset, [0.7, 0.3])
train_loader = th.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_loader = th.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

# Let's create FNP model
encoder = EmbedMetaAttenSeq(
    dim_seq_in=1,
    num_metadata=len(data_obj.idx_dict),
    dim_metadata=1,
    dim_out=60,
    n_layers=2,
    bidirectional=True,
).to(device)
decoder = RegressionSepFNP(
    dim_x=60,
    dim_y=1,
    dim_h=60,
    n_layers=3,
    dim_u=60,
    dim_z=60,
    nodes=len(data_obj.idx_dict),
).to(device)

pre_opt = th.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=PRE_TRAIN_LR
)


def pretrain_epoch(train_it=100):
    losses = []
    pre_opt.zero_grad()
    for i, (r, x, y) in enumerate(train_loader):
        ref_x = r[0, :, :, None].to(device)
        x = x[0, :, :, None].to(device)
        y = y[0, :, None].to(device)
        meta_x = long_tensor(np.arange(ref_x.shape[0]))
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        mean_sample, logstd_sample, log_py, log_pqz, _ = decoder(ref_out_x, out_x, y)
        loss = -(log_py + log_pqz) / x.shape[0]
        print(f'({i}) train: {loss.cpu().item()}')
        loss.backward()
        losses.append(loss.detach().cpu().item())
        if (i + 1) % PRE_BATCH_SIZE == 0:
            pre_opt.step()
            pre_opt.zero_grad()
        if (i + 1) % train_it == 0:
            yield np.mean(losses)
            losses = []
    yield np.mean(losses)


def pre_validate(sample=False, val_it=100):
    losses = []
    while True:
        for i, (r, x, y) in enumerate(val_loader):
            ref_x = r[0, :, :, None].to(device)
            x = x[0, :, :, None].to(device)
            y = y[0, :, None].to(device)
            meta_x = long_tensor(np.arange(ref_x.shape[0]))
            ref_out_x = encoder(ref_x, meta_x)
            out_x = encoder(x, meta_x)
            y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=sample)
            mse_loss = np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2)
            print(mse_loss)
            losses.append(mse_loss)
            if (i + 1) % val_it == 0:
                mse_loss = np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2)
                losses.append(mse_loss)
                yield np.mean(losses)
                losses = []
        yield np.mean(losses)
        losses = []


'''
print("Pretraining...")
pretrain_iterations_per_step = 100
pretrain_counter = 0
train_gen = pretrain_epoch(train_it=pretrain_iterations_per_step)
val_gen = pre_validate(val_it=pretrain_iterations_per_step)

for ep in tqdm(range(PRE_TRAIN_EPOCHS)):
    while True:
        try:
            print("----------------------------------------")
            encoder.train()
            decoder.train()
            loss = next(train_gen)
            print(f"Epoch {ep} loss: {loss}")
            with th.no_grad():
                print("----------------------------------------")
                encoder.eval()
                decoder.eval()
                loss = next(val_gen)
                print(f"Epoch {ep} Val loss: {loss}")
            pretrain_counter += pretrain_iterations_per_step
            th.save(encoder.state_dict(), f'output/custom/encoder_pretrain_{pretrain_counter}_{loss}.pt')
            th.save(decoder.state_dict(), f'output/custom/decoder_pretrain_{pretrain_counter}_{loss}.pt')
        except StopIteration:
            break
'''

corem = Corem(nodes=len(data_obj.idx_dict), c=C, ).to(device)

opt = th.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()) + list(corem.parameters()),
    lr=TRAIN_LR,
)


def jsd_norm(mu1, mu2, var1, var2):
    mu_diff = mu1 - mu2
    t1 = 0.5 * (mu_diff ** 2 + (var1) ** 2) / (2 * (var2) ** 2)
    t2 = 0.5 * (mu_diff ** 2 + (var2) ** 2) / (2 * (var1) ** 2)
    return t1 + t2 - 1.0


def generate_hmatrix():
    ans = np.zeros((len(data_obj.idx_dict), len(data_obj.idx_dict)))
    for i, n in enumerate(data_obj.nodes):
        if len(n.children) == 0:
            ans[n.idx, n.idx] = 1
        c_idx = [x.idx for x in n.children]
        ans[n.idx, c_idx] = 1.0
    return float_tensor(ans)


def jsd_loss(mu, logstd, hmatrix, train_means, train_std):
    lhs_mu = (((mu * train_std + train_means) * hmatrix).sum(1) - train_means) / (
        train_std
    )
    lhs_var = (((th.exp(2.0 * logstd) * (train_std ** 2)) * hmatrix).sum(1)) / (
            train_std ** 2
    )
    ans = th.nan_to_num(jsd_norm(mu, lhs_mu, (2.0 * logstd).exp(), lhs_var))
    return ans.mean()


def train_epoch(train_it=100):
    losses = []
    opt.zero_grad()
    hmatrix = generate_hmatrix()
    th_means = float_tensor(train_means)
    th_std = float_tensor(train_std)
    for i, (r, x, y) in enumerate(train_loader):
        ref_x = r[0, :, :, None].to(device)
        x = x[0, :, :, None].to(device)
        y = y[0, :, None].to(device)
        meta_x = long_tensor(np.arange(ref_x.shape[0]))
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        mean_sample1, logstd_sample1, log_py1, log_pqz, py1 = decoder(ref_out_x, out_x, y)
        mean_sample, logstd_sample, log_py, py = corem(mean_sample1.squeeze(), logstd_sample1.squeeze(), y)
        loss1 = -(log_py + log_pqz) / x.shape[0]
        loss2 = (
                jsd_loss(
                    mean_sample.squeeze(),
                    logstd_sample.squeeze(),
                    hmatrix,
                    th_means,
                    th_std,
                )
                / x.shape[0]
        )
        loss = loss1 + LAMBDA * loss2
        if th.isnan(loss):
            import pdb
            pdb.set_trace()
        print(f'({i}) train: {loss.cpu().item()}, loss1: {loss1.cpu().item()}, loss2: {loss2.cpu().item()}')
        loss.backward()
        losses.append(loss.detach().cpu().item())
        if (i + 1) % PRE_BATCH_SIZE == 0:
            opt.step()
            opt.zero_grad()
        if (i + 1) % train_it == 0:
            yield np.mean(losses)
            losses = []
    yield np.mean(losses)


def validate(sample=False, val_it=100):
    losses = []
    while True:
        for i, (r, x, y) in enumerate(val_loader):
            ref_x = r[0, :, :, None].to(device)
            x = x[0, :, :, None].to(device)
            y = y[0, :, None].to(device)
            meta_x = long_tensor(np.arange(ref_x.shape[0]))
            ref_out_x = encoder(ref_x, meta_x)
            out_x = encoder(x, meta_x)
            y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=sample)
            y_pred, mean_y, logstd_y, _ = corem.predict(mean_y.squeeze(), logstd_y.squeeze(), sample=sample)
            mse_loss = np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2)
            print(mse_loss)
            losses.append(mse_loss)
            if (i + 1) % val_it == 0:
                mse_loss = np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2)
                losses.append(mse_loss)
                yield np.mean(losses)
                losses = []
        yield np.mean(losses)
        losses = []

encoder.load_state_dict(th.load('output/custom/encoder_pretrain_32300_2.161266565322876.pt'))
decoder.load_state_dict(th.load('output/custom/decoder_pretrain_32300_2.161266565322876.pt'))

train_iterations_per_step = 100
train_counter = 0
train_gen = train_epoch(train_it=train_iterations_per_step)
val_gen = validate(val_it=train_iterations_per_step)
print("Training....")
for ep in tqdm(range(TRAIN_EPOCHS)):
    while True:
        try:
            print("----------------------------------------")
            encoder.train()
            decoder.train()
            corem.train()
            loss = next(train_gen)
            print(f"Epoch {ep} loss: {loss}")
            with th.no_grad():
                print("----------------------------------------")
                encoder.eval()
                decoder.eval()
                corem.eval()
                loss = next(val_gen)
                print(f"Epoch {ep} Val loss: {loss}")
            train_counter += train_iterations_per_step
            th.save(encoder.state_dict(), f'output/custom/encoder_train_{train_counter}_{loss}.pt')
            th.save(decoder.state_dict(), f'output/custom/decoder_train_{train_counter}_{loss}.pt')
            th.save(corem.state_dict(), f'output/custom/corem_train_{train_counter}_{loss}.pt')
        except StopIteration:
            break

'''
# Lets evaluate
# One sampple
def sample_data():
    curr_data = train_data.copy()
    encoder.eval()
    decoder.eval()
    corem.eval()
    for t in range(AHEAD):
        ref_x = float_tensor(train_data[:, :, None])
        meta_x = long_tensor(np.arange(ref_x.shape[0]))
        x = float_tensor(curr_data[:, :, None])
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=False)
        y_pred, mean_y, logstd_y, _ = corem.predict(
            mean_y.squeeze(), logstd_y.squeeze(), sample=True
        )
        y_pred = y_pred.cpu().numpy()
        curr_data = np.concatenate([curr_data, y_pred], axis=1)
    return curr_data[:, -AHEAD:]


ground_truth = full_data[:, TRAIN_UPTO: TRAIN_UPTO + AHEAD]
with th.no_grad():
    preds = [sample_data() for _ in tqdm(range(EVAL_SAMPLES))]
preds = np.array(preds)
preds = preds * train_std[:, None] + train_means[:, None]
mean_preds = np.mean(preds, axis=0)
rmse = np.sqrt(np.mean((ground_truth - mean_preds) ** 2))
print(f"RMSE: {rmse}")
crps = ps.crps_ensemble(ground_truth, np.moveaxis(preds, [1, 2, 0], [0, 1, 2])).mean()
print(f"CRPS: {crps}")
'''
