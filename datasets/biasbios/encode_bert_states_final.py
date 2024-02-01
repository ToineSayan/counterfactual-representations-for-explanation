import numpy as np
# from docopt import docopt
import torch
from transformers import BertModel, BertTokenizer
import pickle
from tqdm import tqdm
import json

MAX_TOKENS = 512
ENCODING_WINDOW = 25000


def read_data_file(input_file):
    """
    read the data file with a pickle format
    :param input_file: input path, string
    :return: the file's content
    """
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_lm():
    """
    load bert's language model
    :return: the model and its corresponding tokenizer
    """
    model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return model, tokenizer


def tokenize(tokenizer, data):
    """
    Iterate over the data and tokenize it. Sequences longer than 512 tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: data to tokenize
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    for row in tqdm(data):
        tokens = tokenizer.encode(row['hard_text_untokenized'], add_special_tokens=True, truncation=True)
        # keeping a maximum length of bert tokens: MAX_TOKENS
        tokenized_data.append(tokens[:MAX_TOKENS])
    return tokenized_data


def encode_text(model, data):
    """
    encode the text
    :param model: encoding model
    :param data: data
    :return: one numpy matrix of the data that contains the cls token of each sentence
    """
    all_data_cls = []
    # all_data_avg = []
    batch = []
    for row in tqdm(data):
        batch.append(row)
        input_ids = torch.tensor(batch)
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
            # all_data_avg.append(last_hidden_states.squeeze(0).mean(dim=0).numpy())
            all_data_cls.append(last_hidden_states.squeeze(0)[0].numpy())
        batch = []
    return np.array(all_data_cls)


def save_nested(filename, D):
    tmp = {f'{k}_{kp}': v for k in D.keys() for kp, v in D[k].items()}
    np.savez_compressed(filename, **tmp)


if __name__ == '__main__':
    
    out_dir ='./'
    identifier = 'original' # 'original' or 'CF' (counterfactuals)
    model, tokenizer = load_lm()

    for split in ['train', 'dev', 'test']:
        data = read_data_file(f"{split}.pickle")
        tokens = tokenize(tokenizer, data)

        # Calculation and storage of representations in batches of N observations (to avoid GPU memory issues)
        n_obs = len(data)
        n_windows = int(n_obs/ENCODING_WINDOW) + 1
        for i in range(n_windows):
            cls_data = encode_text(model, tokens[i*ENCODING_WINDOW:min((i+1)*ENCODING_WINDOW,n_obs)])
            # np.save(out_dir + '/' + split + '_avg.npy', avg_data)
            np.save(out_dir + '/' + split + '_' + str(i) + '_cls.npy', cls_data)
        # Complete split reconstruction from saved batches
        cls_data_complete = []
        for i in range(n_windows):
            cls_data_complete.append(np.load(out_dir + '/' + split + '_' + str(i) + '_cls.npy'))
        cls_data_complete = np.concatenate(cls_data_complete, axis=0)
        print(cls_data_complete.shape)
        np.save(out_dir + '/' + split + '_cls.npy', cls_data_complete)

    # Save the full dataset
    dataset = dict()
    for split in  ['train', 'dev', 'test']:
        d = {'train': 'train', 'dev': 'validation', 'test': 'test'}
        data = read_data_file(f"{split}.pickle")
        dataset[d[split]] = dict()
        dataset[d[split]]["X"] =  np.load(out_dir + '/' + split + '_cls.npy')
        dataset[d[split]]["Y"] = np.array([observation["p"] for observation in data])
        dataset[d[split]]["Z_gender"] = np.array([observation["g"] for observation in data])
        save_nested(out_dir + '/' + 'D_' + identifier + '.npz', dataset)


        


