import argparse
import torch
import os
import torch.nn.functional as F
from dataset import HERDataset
from model import STMCL
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
from scipy.stats import pearsonr
import numpy as np
import glob
import json




def get_image_embeddings(model_path, model, test_loader):

    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')
        new_key = new_key.replace('well', 'spot')
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    test_image_embeddings = []
    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_ecode(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)

            spot_feature = batch["expression"].cuda()
            x = batch["position"][:, 0].long().cuda()
            y = batch["position"][:, 1].long().cuda()
            centers_x = model.x_embed(x)
            centers_y = model.y_embed(y)
            spot_embeddings.append(model.spot_projection(spot_feature + centers_x + centers_y))
    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings)



def find(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

    return indices.cpu().numpy()



def save_embeddings(test_dataLoader, args, model_path, save_path, dim):
    os.makedirs(save_path, exist_ok=True)

    model = STMCL(spot_embedding=dim, temperature=1.,
                 image_embedding=1024, projection_dim=256).cuda()

    img_embeddings, spot_embeddings = get_image_embeddings(model_path, model, test_dataLoader)

    img_embeddings = img_embeddings.cpu().numpy()
    spot_embeddings = spot_embeddings.cpu().numpy()

    np.save(os.path.join(save_path, "img_embeddings_" + f"fold_{args.fold}" + ".npy"), img_embeddings.T)
    np.save(os.path.join(save_path, "spot_embeddings_" + f"fold_{args.fold}" + ".npy"), spot_embeddings.T)


def load_data(args):
    if args.dataset == 'her2st':
        print(f'load dataset: {args.dataset}')
        test_dataset = HERDataset(train=False, fold=args.fold)
        test_dataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        # train_dataset = HERDataset(train=True, fold=args.fold)
        # train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # return test_dataLoader, train_dataLoader
        return test_dataLoader


def cal_error(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)  # 样本方差
    std_dev = variance ** 0.5  # 标准差是方差的平方根
    return mean.item(), std_dev.item()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='fold')
    parser.add_argument('--max_fold', type=int, default=8, help='fold')
    parser.add_argument('--dataset', type=str, default='her2st', help='dataset')
    parser.add_argument('--batch_size', type=int, default=48, help='')
    parser.add_argument('--save_embedding', type=bool, default=True, help='')
    parser.add_argument('--result_path', type=str, default=r'model_result_STMCL')
    parser.add_argument('--dim', type=int, default=250)
    args = parser.parse_args()

    with open(r'GT_her2st_250_HEG.json') as f:
        groundtruth = json.load(f)
    
    patients = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    new_groundtruth = {}
    for patient in patients:
        new_groundtruth[patient] = []
        patient_slices = [x for x in list(groundtruth.keys()) if patient in x]
        new_groundtruth[patient] = [np.array(groundtruth[slide]) for slide in patient_slices]
        new_groundtruth[patient] = np.concatenate(new_groundtruth[patient], 0)

    embedding_saved = True

    corr_result = []
    for i in range(args.max_fold):
        args.fold = i
        print("当前fold:", args.fold)
        if not embedding_saved:
            test_dataLoader = load_data(args)
            save_embeddings(test_dataLoader, args, 
                    model_path=glob.glob(os.path.join(args.result_path, args.dataset, f'fold_{args.fold}', 'model_20.pt'))[0],
                    save_path=os.path.join(args.result_path, args.dataset, f'embeddings_{args.fold}'), dim=args.dim)  # 171

        else:
            # save embeddings之后
            image_query = np.load(os.path.join(args.result_path, args.dataset, f'embeddings_{i}', f"img_embeddings_fold_{i}.npy")).T
            
            test_patient = patients[i]
            train_patient = set(np.arange(8)) - {i}
            spot_embeddings = []
            image_embeddings = []
            spot_expression = []

            for j in list(train_patient):
                patient = patients[j]
                save_path=os.path.join(args.result_path, args.dataset, f'embeddings_{j}')
                spot_embeddings.append(np.load(os.path.join(save_path, f"spot_embeddings_fold_{j}.npy")).T)
                image_embeddings.append(np.load(os.path.join(save_path, f"img_embeddings_fold_{j}.npy")).T)
                spot_expression.append(np.array(new_groundtruth[patient]))

            image_key = np.concatenate(image_embeddings)
            spot_key = np.concatenate(spot_embeddings)
            expression_key = np.concatenate(spot_expression)

            indices = find(spot_key, image_query, top_k=800)
            spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
            for i in range(indices.shape[0]):
                a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1)
                reciprocal_of_square_a = np.reciprocal(a ** 2)
                weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
                weights = weights.flatten()
                spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0, weights=weights)
            
            start = 0
            corr_list = []
            test_slices = [x for x in list(groundtruth.keys()) if test_patient in x]
            for test_slice in test_slices:
                expression = np.array(groundtruth[test_slice]).T
                end = np.shape(expression)[1] + start
                pred = spot_expression_pred[start:end].T

                corr = []
                for i in range(250):
                    corr.append(pearsonr(pred[i], expression[i])[0])
                corr_list.append(corr)
                start = end

            corr_ave = np.nanmean(np.array(corr_list), 0)
            corr_result.append(np.mean(corr_ave))
    
    if embedding_saved:
        print(f'corr: {cal_error(corr_result)}')