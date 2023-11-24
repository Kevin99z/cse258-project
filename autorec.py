
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
        self.b = nn.Parameter(torch.zeros(self.num_items))
        
        # Activation functions
        self.activation = nn.Sigmoid()
        self.criterion = nn.MSELoss()

    def forward(self, input_ratings, mask):
        # Encoder
        encoder_output = torch.matmul(input_ratings, self.V) + self.mu
        encoder_activation = self.activation(encoder_output)

        # Decoder
        decoder_output = torch.matmul(encoder_activation, self.W) + self.b
        reconstructed_ratings = decoder_output * mask
        output_R = reconstructed_ratings * self.item_not_seen_mask \
                + self.item_not_seen_score[:input_ratings.shape[0], :]
        return output_R

    def training_step(self, batch, batch_idx):
        input_R, input_mask_R = batch
        input_R, input_mask_R = input_R.to_dense(), input_mask_R.to_dense()
        output_R = self.forward(input_R, input_mask_R)
        # MSE loss
        rec_cost = self.criterion(output_R, input_R)
        pre_reg_cost = torch.norm(self.W)**2 + torch.norm(self.V)**2
        reg_cost = self.lambda_value * 0.5 * pre_reg_cost
        cost = rec_cost + reg_cost
        self.log('train_loss', cost)
        return cost

    def validation_step(self, batch, batch_idx):
        input_R, input_mask_R = batch
        input_R, input_mask_R = input_R.to_dense(), input_mask_R.to_dense()
        output_R = self.forward(input_R, input_mask_R)
        rec_cost = self.criterion(output_R, input_R)
        cost = rec_cost
        self.log('val_loss', cost)
        return cost
    
    def test_step(self, batch, batch_idx):
        input_R, input_mask_R = batch
        input_R, input_mask_R = input_R.to_dense(), input_mask_R.to_dense()
        output_R = self.forward(input_R, input_mask_R)
        rec_cost = self.criterion(output_R, input_R)
        cost = rec_cost
        self.log('test_loss', cost)
        return cost
        

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_neuron', type=int, default=500)
    parser.add_argument('--train_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--optimizer_method', type=str, default='Adam')
    parser.add_argument('--lambda_value', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--display_step', type=int, default=10)
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()
    
    # Initialize data module
    data_module = AutoRecDataModule(
        train_file='train.csv',
        valid_file='valid.csv',
        test_file='test.csv',
        batch_size=args.batch_size
    )

    # Setup data (e.g., create rating and mask matrices)
    data_module.setup()

    # Initialize the AutoRec model
    if args.checkpoint is None:
        model = AutoRec(
            args=args,
            num_users=data_module.num_users,
            num_items=data_module.num_items,
            user_train_set=data_module.user_train_set,
            item_train_set=data_module.item_train_set,
            item_test_set=data_module.item_test_set,
        )
    else:
        model = AutoRec.load_from_checkpoint(args.checkpoint,
                                             args=args,
            num_users=data_module.num_users,
            num_items=data_module.num_items,
            user_train_set=data_module.user_train_set,
            item_train_set=data_module.item_train_set,
            item_test_set=data_module.item_test_set,
                                         )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',     # Metric to monitor
        min_delta=0.00,         # Minimum change in the monitored quantity to qualify as an improvement
        patience=3,             # Number of epochs with no improvement after which training will be stopped
        verbose=False,          # Whether to print logs to stdout
        mode='min',             # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
        )
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpt/',
        filename='{epoch}-{step}',
        every_n_epochs=1,
        monitor='val_loss',
        save_top_k=1,
    )

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=args.train_epoch,
        devices=[0],
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)
    

    # Evaluate the model on the test set
    with torch.no_grad():
        loss_list = []
        for i, data in enumerate(data_module.test_dataloader()):
            loss = model.test_step(data, i)
            loss_list.append(float(loss))
        test_res = sum(loss_list) / len(loss_list)
    print(test_res)

