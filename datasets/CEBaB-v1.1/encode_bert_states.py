import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import json
from datasets import load_dataset

MAX_TOKENS = 128


def read_data_file(split):
    dataset = load_dataset("CEBaB/CEBaB")
    dataset = dataset.rename_column('review_majority', 'labels')
    # dataset = dataset.class_encode_column('labels')
    dataset_split = dataset[split]
    return [obs for obs in dataset_split]

def load_lm(train_set):
    """
    load bert's language model
    :return: the model and its corresponding tokenizer
    """
    model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, f'finetuned_bert_uncased_cebab_{train_set}')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return model, tokenizer


def tokenize(tokenizer, data):
    """
    Iterate over the data and tokenize it. Sequences longer than MAX_TOKENS tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: data to tokenize
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    for row in tqdm(data):
        tokens = tokenizer.encode(row['description'], add_special_tokens=True)
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

    # train_set = 'train_exclusive'
    # train_set = 'train_inclusive'
    train_set = 'train_observational'

    model, tokenizer = load_lm(train_set)

    

    for split in [train_set, 'validation', 'test']:
        # in_file = split2file[split]
        
        data = read_data_file(split)
        print(data[10])
        tokens = tokenize(tokenizer, data)

        cls_data = encode_text(model, tokens)

        np.save(out_dir + '/' + split + '_cls.npy', cls_data)

    
    
    dataset = dict()
    for split in  [train_set, 'validation', 'test']:
        # in_file = split2file[split]
        data = read_data_file(split)
        dataset[split] = dict()
        dataset[split]["X"] =  np.load(out_dir + '/' + split + '_cls.npy')
        dataset[split]["Y"] = np.array([observation["labels"] for observation in data])
        dataset[split]["Z_food"] = np.array([observation["food_aspect_majority"] for observation in data])
        dataset[split]["Z_ambiance"] = np.array([observation["ambiance_aspect_majority"] for observation in data])
        dataset[split]["Z_service"] = np.array([observation["service_aspect_majority"] for observation in data])
        dataset[split]["Z_noise"] = np.array([observation["noise_aspect_majority"] for observation in data])

    save_nested(out_dir + '/' + 'D_' + train_set + '.npz', dataset)



        


