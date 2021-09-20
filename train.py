import os
import sys
import random
import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model import *
from tokenizer import *

def progressLearning(value, endvalue, loss, acc, bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}/{2} \t Loss : {3:.3f}, Acc : {4:.3f}".format(arrow + spaces, 
        value+1, 
        endvalue, 
        loss, 
        acc)
    )
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args) :
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Text Data
    print('Load Data')
    text_data = load_data(args.data_dir)

    # -- Tokenize & Encoder
    kor_text_path = os.path.join(args.token_dir, 'dialogue.txt')
    if os.path.exists(kor_text_path) == False :
        write_data(text_data, kor_text_path, preprocess_kor)
        train_spm(args.token_dir,  'dialogue.txt', 'kor_tokenizer' , args.token_size)
    kor_tokenizer = get_spm(args.token_dir, 'kor_tokenizer.model')

    print('Encode Data')
    idx_data = []
    for sen in tqdm(text_data) :
        sen = preprocess_kor(sen)
        idx_list = kor_tokenizer.encode_as_ids(sen)
        idx_data.append(idx_list)

    # -- Dataset
    ngram_dset = NgramDataset(args.token_size, args.window_size)
    cen_data, con_data = ngram_dset.get_data(idx_data)
    word2vec_dset = Word2VecDataset(cen_data, con_data)

    # -- DataLoader
    data_loader = DataLoader(word2vec_dset,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        batch_size=args.batch_size
    )
    
    # -- Model
    model_module = getattr(import_module('model') , args.model) 
    model = model_module(args.embedding_size, args.token_size).to(device)

    # -- Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -- Scheudler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- Loss
    criterion = nn.CrossEntropyLoss().to(device)

    # -- Training
    log_count = 0
    for epoch in range(args.epochs) :
        idx = 0
        mean_loss = 0.0
        mean_acc = 0.0
        model.train()
        print('Epoch : %d/%d \t Learning Rate : %e' %(epoch, args.epochs, optimizer.param_groups[0]["lr"]))
        for center, context in data_loader :
            center = center.long().to(device)
            context = context.long().to(device)

            if args.model == 'CBOW' :
                in_data = context
                out_label = center
            else : 
                in_data = center.unsqueeze(1).repeat(1,args.window_size-1)
                in_data = in_data.view([-1,])
                out_label = context.view([-1,])
            
            out_data = model(in_data)

            loss = criterion(out_data, out_label)
            acc = (torch.argmax(out_data,-1) == out_label).float().mean()
            loss.backward()
            optimizer.step()

            mean_loss += loss
            mean_acc += acc

            progressLearning(idx, len(data_loader), loss.item(), acc.item())
            if (idx + 1) % 10 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                writer.add_scalar('train/acc', acc.item(), log_count)
                log_count += 1
            idx += 1

        mean_loss /= len(data_loader)
        mean_acc /= len(data_loader)

        torch.save({'epoch' : (epoch) ,  
            'model_state_dict' : model.state_dict() , 
            'loss' : mean_loss.item(), 
            'acc' : mean_acc.item()},
        os.path.join(args.model_dir,'word2vec_model.pt'))

        scheduler.step()
        print('\nMean Loss : %.3f , Mean Acc : %.3f\n' %(mean_loss, mean_acc))

    kor_weight = model.get_weight()
    kor_weight = kor_weight.detach().cpu().numpy()
    kor_weight[0] = 0.0

    np.save(os.path.join(args.embedding_dir, 'kor_' + args.model.lower() + '.npy'), kor_weight)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--token_size', type=int, default=32000, help='number of bpe merge (default: 32000)')
    parser.add_argument('--model', type=str, default='CBOW', help='model of embedding (default: CBOW)')
    parser.add_argument('--embedding_size', type=int, default=512, help='embedding size of token (default: 512)')
    parser.add_argument('--window_size', type=int, default=13, help='window size (default: 13)')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')    

    parser.add_argument('--data_dir', type=str, default='./Data', help = 'data path')
    parser.add_argument('--token_dir', type=str, default='./Token' , help='token data dir path')
    parser.add_argument('--log_dir', type=str, default='./Log' , help='loggind data dir path')
    parser.add_argument('--embedding_dir', type=str, default='./Embedding' , help='embedding dir path')
    parser.add_argument('--model_dir', type=str, default='./Model' , help='best model dir path')

    args = parser.parse_args()

    train(args)
