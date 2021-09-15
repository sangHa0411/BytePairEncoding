from os import pardir
import re
import random
import collections
import pandas as pd
from tqdm import tqdm
from enum import IntEnum

class Token(IntEnum) :
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

def kor_preprocess(sen) :
    sen = re.sub('[0-9]+,*[0-9]*', 'NUM ', sen) # convert digit to NUM token
    sen = re.sub('[^A-Z가-힣!?.,\']', ' ' , sen) # filter character except alphabet uppercase, korean characters, punctuation 
    sen = re.sub(' {2,}', ' ', sen) # merge space 
    return sen

def en_preprocess(sen) :
    sen = sen.lower() # make lower case
    re.sub('[0-9]+,*[0-9]*', 'NUM ', sen) # convert digit to NUM token 
    sen = re.sub('[^a-zA-Z!?.,\']' , ' ' , sen) # filter character except alphabet, punctuation
    sen = re.sub(' {2,}' , ' ' , sen) # merge space
    return sen

class Preprocessor :
    def __init__(self, text, preprocess, tokenize) :
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.data = [tokenize(preprocess(sen)) for sen in text]

    # build data for bpe
    def get_bpe(self, merge_count) :
        counter = collections.Counter()
        for tok_list in self.data :
            counter.update(tok_list)
        counter = dict(counter)
        
        bpe_dict = {}
        subword_set = set()
        for tok, counts in counter.items() :
            ch_tuple = tuple(tok) + ('_',)
            ch_str = ' '.join(ch_tuple)
            bpe_dict[ch_str] = counts
            subword_set.update(list(ch_tuple))

        subword_list = list(subword_set)

        bpe_code = {}
        for i in tqdm(range(merge_count)) :
            pairs = self.get_stats(bpe_dict)
            if len(pairs) == 0 :
                break
            best = max(pairs, key=pairs.get)
            bpe_dict = self.merge_vocab(best, bpe_dict)
            bpe_code[best] = i
            bigram = ''.join(best)
            subword_list.append(bigram)

        subword_list = [Token.PAD, Token.UNK, Token.SOS, Token.EOS] + sorted(subword_list, key=len, reverse=True)
        return bpe_code, subword_list

    # code from paper
    # count frequency of bigram sub word which has to merge
    # the more bigram sub word appears, the more possible it has meaning
    def get_stats(self, vocab):
        pairs = collections.defaultdict(int) 
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i],symbols[i+1]] += freq 
        return pairs

    # code from paper
    # merge 2 sub word and make bigram for every tok in dict(v_in)
    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') 
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word] 
        return v_out

class Tokenizer:
    def __init__(self, bpe_code, preprocess, tokenize) :
        if isinstance(bpe_code, pd.DataFrame) :
            self.set_data(bpe_code)
        else :
            self.bpe_code = bpe_code
        self.preprocess = preprocess
        self.tokenize_fn = tokenize

    # get sub word bigram from word
    # word_ -> (w,o), (o,r), (r,d), (d,_) 
    def get_pairs(self, word) :
        pairs = set()
        prev_char = word[0]
        for char in word[1:] :
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    # code from wikidocs
    def get_sub(self, orig) :
        word = tuple(orig) + ('_',) # split word to subword tuple
        pairs = self.get_pairs(word) # get sub word bigram from character tuple

        if not pairs:
            return orig

        iteration = 0
        # for each iteration
        while True:
            iteration += 1
            # get candidate subword bigram which will be merged
            # key : sub word bigram tuple
            # value : index / the less the value, the more frequent subword token
            bigram = min(pairs, key = lambda pair: self.bpe_code.get(pair, float('inf')))

            if bigram not in self.bpe_code:
                break
            # first subword, second subword
            first, second = bigram 
            new_word = []
            i = 0
            while i < len(word):
                try:
                    # i is smaller than index of first subword in subword tuple
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j # get first token index
                except:
                    # fir tok is not in subword touple which index is larger then i
                    new_word.extend(word[i:])
                    break
                 # meet first subword, second subword in subword tuple
                if (word[i] == first) and (i < len(word)-1) and (word[i+1] == second):
                    new_word.append(first+second) # merge 2 subword
                    i += 2
                # add first tok in word list
                else: 
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word) # convert list to tuple
            word = new_word
            if len(word) == 1: # merage all character
                break # break loop
            else:
                pairs = self.get_pairs(word) # get sub word bigram from merged character tuple
                
        return word

    def set_data(self, bpe_data) :
        bpe_code = {}
        data_list = list(bpe_data['data'])
        count_list = list(bpe_data['count'])
        for i in range(len(bpe_data)) :
            tok_list = re.findall('\'[a-zA-Z가-힣!?.,\'_]+\'', data_list[i])
            tok_tuple = tuple([tok[1:-1] for tok in tok_list])
            count = count_list[i]
            bpe_code[tok_tuple] = count
        self.bpe_code = bpe_code

    def get_data(self) :
        return self.bpe_code

    def tokenize(self, sen) :
        sen = self.preprocess(sen) # preprocess sen
        tok_list = self.tokenize_fn(sen) # tokenize sen using nltk or mecab
        subword_list = []
        for tok in tok_list : # for each tok
            subwords = self.get_sub(tok) # get subword list
            subword_list += list(subwords) # extend subword list
            
        return subword_list

class Encoder :
    def __init__(self, subword_list) :
        if isinstance(subword_list, pd.DataFrame) :
            self.set_data(subword_list)
        else :
            self.sub2idx = self.build_data(subword_list)

    def build_data(self, subword_list) :
        idx = 0
        sub2idx = {}
        for tok in subword_list :
            sub2idx[tok] = idx
            idx += 1
        return sub2idx

    def get_data(self) :
        return self.sub2idx

    def set_data(self, data_df) :
        data_size = len(data_df)
        tok_list = list(data_df['token'])
        idx_list = list(data_df['index'])
        sub2idx = {}
        for i in range(data_size) :
            sub2idx[tok_list[i]] = idx_list[i]
        self.sub2idx = sub2idx

    def encode(self, tok_list) :
        idx_list = [] 
        for tok in tok_list :
            if tok in self.sub2idx :
                idx = self.sub2idx[tok]
            else :
                idx = Token.UNK
            idx_list.append(idx)
        return idx_list
