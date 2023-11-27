# %%
import os
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
import torch.optim as optim
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping

class AutoRecDataset(Dataset):
    def __init__(self, R, mask_R):
        self.R = R
        self.mask_R = mask_R

    def __len__(self):
        return len(self.R)

    def __getitem__(self, idx):
        return self.R[idx], self.mask_R[idx]

class AutoRec(LightningModule):
    def __init__(self, args, num_users, num_items, user_train_set, item_train_set, item_test_set):
        """
        Initializes the AutoRec model.

        Args:
            args: A namespace or dictionary containing the model hyperparameters.
            num_users: The number of users in the dataset.
            num_items: The number of items in the dataset.
            R: The complete user-item rating matrix.
            mask_R: A binary matrix of the same shape as R, where 1 indicates a known rating.
            C: Confidence matrix, used in some recommendation models (not utilized in this code).
            train_R: The user-item rating matrix for training.
            train_mask_R: A binary matrix for training data, where 1 indicates a known rating.
            test_R: The user-item rating matrix for testing.
            test_mask_R: A binary matrix for testing data, where 1 indicates a known rating.
            num_train_ratings: The number of ratings in the training set.
            num_test_ratings: The number of ratings in the testing set.
            user_train_set: Set of users that have ratings in the training set.
            item_train_set: Set of items that have ratings in the training set.
            user_test_set: Set of users that have ratings in the testing set.
            item_test_set: Set of items that have ratings in the testing set.
            result_path: Path to save the training and testing records.
        """
        super(AutoRec, self).__init__()
        # self.save_hyperparameters()

        # Model hyperparameters
        self.hidden_neurons = args.hidden_neuron
        self.train_epochs = args.train_epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.base_lr
        self.optimizer_method = args.optimizer_method
        self.lambda_value = args.lambda_value
        self.grad_clip = args.grad_clip
        self.base_lr = args.base_lr

        # Data information
        self.num_users = num_users
        self.num_items = num_items
        # self.train_R = train_R
        # self.train_mask_R = train_mask_R
        # self.test_R = test_R
        # self.test_mask_R = test_mask_R
        # self.num_train_ratings = num_train_ratings
        # self.num_test_ratings = num_test_ratings
        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.item_test_set = item_test_set
        self.item_not_seen = item_test_set - item_train_set
        # indices vector of item_not_seen
        item_not_seen_score = torch.zeros((self.batch_size, self.num_items))
        item_not_seen_mask = torch.ones(self.num_items)
        for idx in self.item_not_seen:
            item_not_seen_mask[idx] = 0
            item_not_seen_score[:, idx] = 4.41
        self.register_buffer('item_not_seen_score', item_not_seen_score)
        self.register_buffer('item_not_seen_mask', item_not_seen_mask)

        # self.user_test_set = user_test_set
        # self.item_test_set = item_test_set

        # Model components
        self.V = nn.Parameter(torch.randn(self.num_items, self.hidden_neurons) * 0.03)
        self.W = nn.Parameter(torch.randn(self.hidden_neurons, self.num_items) * 0.03)
        self.mu = nn.Parameter(torch.zeros(self.hidden_neurons))
        self.b = nn.Parameter(torch.zeros(self.num_items)+4)
        
        # Activation functions
        self.activation = nn.Sigmoid()
        self.criterion = nn.MSELoss(reduction='sum')
        self.errors = []

    def forward(self, input_ratings, mask):
        # Encoder
        encoder_output = torch.matmul(input_ratings, self.V) + self.mu
        encoder_activation = self.activation(encoder_output)

        # Decoder
        decoder_output = torch.matmul(encoder_activation, self.W) + self.b
        reconstructed_ratings = decoder_output
        output_R = reconstructed_ratings * self.item_not_seen_mask \
                + self.item_not_seen_score[:input_ratings.shape[0], :]
        return output_R * mask

    def training_step(self, batch, batch_idx):
        input_R, input_mask_R = batch
        input_R, input_mask_R = input_R.to_dense(), input_mask_R.to_dense()
        output_R = self.forward(input_R, input_mask_R)
        # MSE loss
        rec_cost = self.criterion(output_R, input_R) / input_mask_R.sum()
        pre_reg_cost = torch.norm(self.W)**2 + torch.norm(self.V)**2
        reg_cost = self.lambda_value * 0.5 * pre_reg_cost
        cost = rec_cost + reg_cost
        self.log('train_loss', rec_cost, batch_size=input_mask_R.sum(), prog_bar=True)
        return cost

    def validation_step(self, batch, batch_idx):
        input_R, input_mask_R = batch
        input_R, input_mask_R = input_R.to_dense(), input_mask_R.to_dense()
        output_R = self.forward(input_R, input_mask_R)
        rec_cost = self.criterion(output_R, input_R) / input_mask_R.sum()
        cost = rec_cost
        self.log('val_loss', cost, batch_size=input_mask_R.sum())
        return cost
    
    def test_step(self, batch, batch_idx):
        input_R, input_mask_R = batch
        input_R, input_mask_R = input_R.to_dense(), input_mask_R.to_dense()
        output_R = self.forward(input_R, input_mask_R)
        # collect all the errors in a list
        error_R = torch.abs(output_R)
        errors = error_R[input_mask_R == 1]
        self.errors.extend(errors.cpu().tolist())
        loss = self.criterion(output_R, input_R)
        # cost = rec_cost
        # self.log('test_loss', cost, batch_size=input_mask_R.sum())
        return loss, input_mask_R.sum()
        

    def configure_optimizers(self):
        if self.optimizer_method == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.base_lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = optim.RMSprop(self.parameters(), lr=self.base_lr)
        else:
            raise ValueError("Optimizer Key ERROR")
        
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        return [optimizer], [scheduler]



# Assuming that the AutoRec and AutoRecDataset classes have been defined as above

class AutoRecDataModule(LightningDataModule):
    def __init__(self, train_file, valid_file, test_file, batch_size):
        super().__init__()
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load data
        train_df = pd.read_csv(self.train_file)
        valid_df = pd.read_csv(self.valid_file)
        test_df = pd.read_csv(self.test_file)
        self.user_train_set = set(train_df['user'].unique())
        self.item_train_set = set(train_df['item'].unique())
        self.item_test_set = set(test_df['item'].unique())

        # Get the number of users and items
        self.num_users = max(train_df['user'].max(), valid_df['user'].max(), test_df['user'].max()) + 1
        self.num_items = max(train_df['item'].max(), valid_df['item'].max(), test_df['item'].max()) + 1

        # Create rating and mask matrices
        def create_matrix(df, num_users=self.num_users, num_items=self.num_items):
            users = df['user'].values
            items = df['item'].values
            ratings = df['rating'].values

            # Create index and value tensors for the sparse matrix
            indices = torch.LongTensor([users, items])  # 2 x nnz matrix
            values = torch.FloatTensor(ratings)  # 1 x nnz matrix
            
            # Create a sparse tensor
            shape = (num_users, num_items)
            ratings = torch.sparse_coo_tensor(indices, values, torch.Size(shape), dtype=torch.float32)
            mask_values = torch.ones_like(values)
            mask = torch.sparse_coo_tensor(indices, mask_values, torch.Size(shape), dtype=torch.float32)
            
            return ratings.cuda(), mask.cuda()

        self.train_ratings, self.train_mask = create_matrix(train_df)
        self.valid_ratings, self.valid_mask = create_matrix(valid_df)
        self.test_ratings, self.test_mask = create_matrix(test_df)

    def train_dataloader(self):
        dataset = TensorDataset(self.train_ratings, self.train_mask)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = TensorDataset(self.valid_ratings, self.valid_mask)
        return DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        dataset = TensorDataset(self.test_ratings, self.test_mask)
        return DataLoader(dataset, batch_size=self.batch_size)
#%%
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_neuron', type=int, default=500)
    parser.add_argument('--train_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--optimizer_method', type=str, default='Adam')
    parser.add_argument('--lambda_value', type=float, default=1)
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--display_step', type=int, default=10)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--f')
    args = parser.parse_args()
    return args
args = parse_args()


# %%
data_module = AutoRecDataModule(
    train_file='train.csv',
    valid_file='valid.csv',
    test_file='test.csv',
    batch_size=args.batch_size
)
device = 'cuda:1'
data_module.setup()
model = AutoRec.load_from_checkpoint("ckpt/lr=0.001-lambda=0.001-epoch=47.ckpt",
                                     args=args,
                    num_users=data_module.num_users,
                    num_items=data_module.num_items,
                    user_train_set=data_module.user_train_set,
                    item_train_set=data_module.item_train_set,
                    item_test_set=data_module.item_test_set,
        ).to(device)
                                     
# %%
# Evaluate the model on the test set
with torch.no_grad():
    total_loss, cnt = 0, 0
    for i, data in enumerate(data_module.test_dataloader()):
        data = [d.to(device) for d in data]
        loss, size = model.test_step(data, i)
        total_loss += float(loss)
        cnt += int(size)
    test_res = total_loss / cnt
print(test_res)
# %%
import matplotlib.pyplot as plt
errors = model.errors
plt.hist(errors, bins=25, range=(0, 5))
# %%
