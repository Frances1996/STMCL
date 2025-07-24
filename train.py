import argparse
import torch
import os
import torch.nn.functional as F
from dataset import HERDataset
from model import STMCL
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AvgMeter, get_lr
import os
import torch
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_epochs', type=int, default=90, help='')
parser.add_argument('--fold', type=int, default=0, help='fold')
parser.add_argument('--max_fold', type=int, default=8, help='fold')
parser.add_argument('--temperature', type=float, default=1., help='temperature')
parser.add_argument('--dim', type=int, default=250, help='spot_embedding dimension (# HVGs)')  # 171, 785, 58, 50
parser.add_argument('--image_embedding_dim', type=int, default=1024, help='image_embedding dimension')
parser.add_argument('--projection_dim', type=int, default=256, help='projection_dim ')
parser.add_argument('--dataset', type=str, default='her2st', help='dataset')  # Mouse_spleen


def load_data(args):
    if args.dataset == 'her2st':
        print(f'load dataset: {args.dataset}')
        train_dataset = HERDataset(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = HERDataset(train=False, fold=args.fold)
        test_dataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_dataLoader, test_dataLoader
    
        datasets.pop(args.fold)
        print("Test name: ", examples[args.fold], "Test fold: ", args.fold)

        train_dataset = torch.utils.data.ConcatDataset(datasets)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        return train_loader, examples


def train(model, train_dataLoader, optimizer, epoch):
    model.train()
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
    for batch in tqdm_train:
        batch = {k: v.cuda() for k, v in batch.items() if
                 k == "image" or k == "expression" or k == "position"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_train.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), epoch=epoch)


def save_model(args, model, best_mse=None, epoch=None):
    os.makedirs(rf"model_result_STMCL/{args.dataset}/fold_{args.fold}", exist_ok=True)
    if best_mse:
        torch.save(model.state_dict(), rf"model_result_STMCL/{args.dataset}/fold_{args.fold}/best_mse_{best_mse:.5f}.pt")
    else:
        torch.save(model.state_dict(), rf"model_result_STMCL/{args.dataset}/fold_{args.fold}/model_{epoch}.pt")



def test(model, test_dataloader):
    model.eval()
    loss_list = []
    with torch.no_grad():
        tqdm_test = tqdm(test_dataloader, total=len(test_dataloader))

        for batch in tqdm_test:
            batch = {k: v.cuda() for k, v in batch.items() if
                    k == "image" or k == "expression" or k == "position"}
            loss = model(batch)
            loss_list.append(loss.item())
            del batch
    
    torch.cuda.empty_cache()
    return(np.mean(loss_list))
# 因为效果太差所以需要保存模型的机制


def main():
    args = parser.parse_args()
    for i in range(8):
        args.fold = i
        print("当前fold:", args.fold)
        train_dataLoader, test_dataloader = load_data(args)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = STMCL(spot_embedding=args.dim, temperature=args.temperature,
                     image_embedding=args.image_embedding_dim, projection_dim=args.projection_dim).cuda()
        
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, weight_decay=1e-3
        )
        best_mse = torch.inf
        pre_best_path = ''
        first_best = True
        

        for epoch in range(args.max_epochs):

            train(model, train_dataLoader, optimizer, epoch)

            mse = test(model, test_dataloader)
            if mse < best_mse:
                best_mse = mse
                save_model(args, model, best_mse=best_mse)
                print(f"Saved Model, best mse: {mse}; epoch: {epoch}")
                if first_best == False:
                    os.remove(pre_best_path)
                pre_best_path = f"model_result_STMCL/{args.dataset}/fold_{args.fold}/best_mse_{best_mse:.5f}.pt"
                first_best = False

            if epoch % 10 == 0:
                save_model(args, model, epoch=epoch)


if __name__ == "__main__":
    main()
