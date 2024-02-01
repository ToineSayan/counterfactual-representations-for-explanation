import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import json

MAX_TOKENS = 256


def read_data_file(json_input_file, identifier, split, read_templates_only = False):
    """
    read the data file with a pickle format
    :param input_file: input path, string
    :return: the file's content
    """
    data = []
    with open(json_input_file, 'rb') as f:
        for observation in f:
            o = json.loads(observation)
            if o["split"] == split:
                if read_templates_only:
                    o[identifier]["sentence"] = o["template"]
                data.append(o[identifier])
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
    Iterate over the data and tokenize it. Sequences longer than MAX_TOKENS tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: data to tokenize
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    for row in tqdm(data):
        tokens = tokenizer.encode(row['sentence'], add_special_tokens=True)
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

    
    in_file = 'new_data.json'
    out_dir ='./'
    
    for identifier in ['balanced', 'aggressive_gender','aggressive_race', 'CF_gender', 'CF_race', 'templates']:
        
        read_templates_only = False # will replace the sentences by their raw template with placeholders
        if identifier == 'templates':
             read_templates_only = True
             identifier = 'balanced'


        model, tokenizer = load_lm()

        

        for split in ['train', 'validation', 'test']:
            data = read_data_file(in_file, identifier, split, read_templates_only=read_templates_only)
            tokens = tokenize(tokenizer, data)

            cls_data = encode_text(model, tokens)

            np.save(out_dir + '/' + split + '_cls.npy', cls_data)


            Z_gender = np.array([observation["gender_label"] for observation in data])
            Z_race = np.array([observation["race_label"] for observation in data])
        
        
        dataset = dict()
        for split in  ['train', 'validation', 'test']:
            data = read_data_file(in_file, identifier, split, read_templates_only=read_templates_only)
            dataset[split] = dict()
            dataset[split]["X"] =  np.load(out_dir + '/' + split + '_cls.npy')
            dataset[split]["Y"] = np.array([observation["poms_label"] for observation in data])
            dataset[split]["Z_gender"] = np.array([observation["gender_label"] for observation in data])
            dataset[split]["Z_race"] = np.array([observation["race_label"] for observation in data])
        if read_templates_only:
            identifier = identifier + '_' + 'templates'
        save_nested(out_dir + '/' + 'D_' + identifier + '.npz', dataset)



        


