import pandas as pd
import re
import os

def load_bpe_code(bpe_code_path) :
    bpe_code_df = pd.read_csv(bpe_code_path)
    bpe_code = {}
    for i in range(len(bpe_code_df)) :
        tok_list = re.findall('\'[a-zA-Z가-힣!?.,\'<>/]+\'' , bpe_code_df['bigram'][i])
        tok_tuple = tuple([tok[1:-1] for tok in tok_list])
        bpe_code[tok_tuple] = bpe_code_df['count'][i]
    return bpe_code

def save_bpe_code(bpe_code, bpe_code_path) :
    bpe_code_df = pd.DataFrame({'bigram' : list(bpe_code.keys()) ,
        'count' : list(bpe_code.values())}
    )
    bpe_code_df.to_csv(bpe_code_path)

def load_bpe_dict(bpe_dict_path) :
    bpe_dict_df = pd.read_csv(bpe_dict_path)
    bpe_dict = {}
    for i in range(len(bpe_dict_df)) :
        tok = bpe_dict_df['token'][i]
        count = bpe_dict_df['count'][i]
        bpe_dict[tok] = count
    return bpe_dict

def save_bpe_dict(bpe_dict, bpe_dict_path) :
    bpe_dict_df = pd.DataFrame({'token' : list(bpe_dict.keys()) ,
        'count' : list(bpe_dict.values())}
    )
    bpe_dict_df.to_csv(bpe_dict_path)

def load_sub2idx(sub2idx_path) :
    sub2idx_df = pd.read_csv(sub2idx_path)
    sub2idx_dict = {}
    idx2sub_dict = {}
    for i in range(len(sub2idx_df)) :
        sub = sub2idx_df['sub'][i]
        idx = sub2idx_df['idx'][i]

        sub2idx_dict[sub] = idx
        idx2sub_dict[idx] = sub

    return sub2idx_dict, idx2sub_dict
    
def save_sub2idx(sub2idx, sub2idx_path) :
    sub2idx_df = pd.DataFrame({'sub' : list(sub2idx.keys()) ,
        'idx' : list(sub2idx.values())}
    )
    sub2idx_df.to_csv(sub2idx_path)

