import json
import os.path
from collections import Counter

import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import datasets
import torch
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, AutoTokenizer


class Summarization_Dataset(Dataset):
    def __init__(self, dataset, max_seq_len_sqrt, sent_encoder_type='sbert', summary_type='oracle', summary_level='sen'):
        super(Summarization_Dataset, self).__init__()
        self.max_seq_len_sqrt = max_seq_len_sqrt
        self.dataset = dataset
        self.length = len(self.dataset['data'])
        self.summary_type = summary_type
        self.summary_level = summary_level

        self.sent_encoder_type = sent_encoder_type
        if self.sent_encoder_type == 'sbert':
            self.sen_encoder = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.sen_encoder = BertModel.from_pretrained("bert-base-uncased")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            article = self.dataset['data'][idx]['article']
            summary = self.dataset['data'][idx]['summary']
            sen_labels = self.dataset['data'][idx]['label']
            cand_sens =  self.dataset['data'][idx]['cand_sens']
            cand_combs = self.dataset['data'][idx]['cand_combs']


            article_sens = article
            summary_sens = summary

            if self.sent_encoder_type == 'sbert':
                article_sen_embeds = self.sen_encoder.encode(article_sens, convert_to_tensor=True).cpu()
                if self.summary_type == 'oracle': # ext_setting
                    if self.summary_level == 'sum':
                        summary_sens_input = ".".join([article_sens[i] for i in sen_labels])
                        summary_sen_embeds = self.sen_encoder.encode(summary_sens_input, convert_to_tensor=True).unsqueeze(0).cpu()
                    elif self.summary_level == 'sen':
                        summary_sen_embeds = torch.stack([article_sen_embeds[i, :] for i in sen_labels])

                elif self.summary_type == 'ref':
                    if self.summary_level == 'sum':
                        summary_sens_input = ".".join(summary_sens)
                    elif self.summary_level == 'sen':
                        summary_sens_input = summary_sens
                    summary_sen_embeds = self.sen_encoder.encode(summary_sens_input, convert_to_tensor=True).cpu()

            elif self.sent_encoder_type == 'bert':
                inputs = self.tokenizer(article_sens, return_tensors="pt", padding=True, truncation=True)
                article_sen_embeds = self.sen_encoder(**inputs).last_hidden_state[:, 0, :] #(sen_num, word_num, emb_dim)

                if self.summary_type == 'oracle': # ext_setting
                    if self.summary_level == 'sum':
                        summary_sens_input = ".".join([article_sens[i] for i in sen_labels])
                        inputs = self.tokenizer(summary_sens_input, return_tensors="pt", padding=True, truncation=True)
                        summary_sen_embeds = self.sen_encoder(**inputs).last_hidden_state[:, 0, :]
                    elif self.summary_level == 'sen':
                        summary_sen_embeds = torch.stack([article_sen_embeds[i, :] for i in sen_labels])

                elif self.summary_type == 'ref':
                    if self.summary_level == 'sum':
                        summary_sens_input = ".".join(summary_sens)
                    elif self.summary_level == 'sen':
                        summary_sens_input = summary_sens
                    inputs = self.tokenizer(summary_sens_input, return_tensors="pt", padding=True, truncation=True)
                    summary_sen_embeds = self.sen_encoder(**inputs).last_hidden_state[:, 0, :]



            return [], {'article_sen_embeds': article_sen_embeds,
                        'summary_sen_embeds': summary_sen_embeds,
                        'article_sens': article_sens,
                        'summary_sens': summary_sens,
                        'label': sen_labels,
                        'cand_sens': cand_sens,
                        'cand_combs': cand_combs}



def get_dataset(article_corpus, summary_corpus, label, ext_idx, cand_combs):

    text_dataset = datasets.Dataset.from_dict({'summary': summary_corpus,
                                               'article': article_corpus,
                                               'label': label,
                                               'cand_sens': ext_idx,
                                               'cand_combs': cand_combs})
    dataset = datasets.DatasetDict()
    dataset['data'] = text_dataset
    return dataset




def load_data_summarization(batch_size, image_size, data_name='cnn',
                            split=None, shuffle=True, sent_encoder_type='sbert', summary_type='oracle', summary_level='sen'):
    # image size is the sqrt of max seq len
    max_seq_len = image_size ** 2
    def collect_fn(data):
        article_sen_embeds = [d[1]['article_sen_embeds'] for d in data]
        summary_sen_embeds = [d[1]['summary_sen_embeds'] for d in data]
        article_sens = [d[1]['article_sens'] for d in data]
        summary_sens = [d[1]['summary_sens'] for d in data]
        labels = [d[1]['label'] for d in data]
        cand_sens = [d[1]['cand_sens'] for d in data]
        cand_combs = [d[1]['cand_combs'] for d in data]

        if data_name.startswith('cnn') or data_name.startswith('xsum') or data_name.startswith('reddit'):
            max_article_len = 45
            max_sum_len = 3
        elif data_name.startswith('pubmed'):
            max_article_len = 120
            max_sum_len = 8
        elif data_name.startswith('wikihow'):
            max_article_len = 60
            max_sum_len = 4
        elif  data_name.startswith('multinews'):
            max_article_len = 55
            max_sum_len = 9

        article_sen_masks = torch.zeros(len(article_sen_embeds), max_article_len)
        article_sen_embeds_padded = torch.zeros((len(article_sen_embeds), max_article_len , article_sen_embeds[0].shape[1]))
        for i, embs in enumerate(article_sen_embeds):
            if embs.shape[0] > max_article_len:
                article_sen_embeds_padded[i, :, :] = embs[:max_article_len, :]
                article_sen_masks[i, :] = 1
            else:
                article_sen_embeds_padded[i, 0:embs.shape[0], :] = embs
                article_sen_masks[i, 0:embs.shape[0]] = 1

        summary_sen_masks = torch.zeros(len(summary_sen_embeds), max_sum_len)
        summary_sen_embeds_padded = torch.zeros((len(summary_sen_embeds), max_sum_len, summary_sen_embeds[0].shape[1]))
        for i, embs in enumerate(summary_sen_embeds):
            if embs.shape[0] > max_sum_len:
                summary_sen_embeds_padded[i, :, :] = embs[:max_sum_len, :]
                summary_sen_masks[i, :] = 1
            else:
                summary_sen_embeds_padded[i, 0:embs.shape[0], :] = embs
                summary_sen_masks[i, 0:embs.shape[0]] = 1

        contra_labels = torch.zeros(len(labels), max_article_len + max_sum_len)
        ce_labels = torch.zeros(len(labels), max_sum_len) - 1
        for i, label_idx in enumerate(labels):
            summary_len = min(len(label_idx), max_sum_len)
            # ce_labels[i, summary_len:] = max_article_len
            for j, idx in enumerate(label_idx):
                if idx < max_article_len and j < max_sum_len:
                    ce_labels[i, j] = idx
                    contra_labels[i, idx] = j + 1
                    contra_labels[i, max_article_len + j] = j + 1
                if j > summary_len:
                    break

        src_mask = torch.zeros(((max_article_len + max_sum_len), (max_article_len + max_sum_len)))
        src_mask[:max_article_len, max_article_len:] = float('-inf')

        article_sen_masks = 1 - article_sen_masks
        summary_sen_masks = 1 - summary_sen_masks

        return torch.zeros((len(summary_sen_embeds), max_seq_len, 128)), {'article_sen_embeds': article_sen_embeds_padded,
                                           'article_sen_masks': article_sen_masks.bool(),
                                           'summary_sen_embeds': summary_sen_embeds_padded,
                                           'summary_sen_masks': summary_sen_masks.bool(),
                                            'article_sens': article_sens,
                                            'summary_sens': summary_sens,
                                            'label': ce_labels,
                                            'cand_sens': cand_sens,
                                            'cand_combs': cand_combs,
                                            'contra_labels': contra_labels,
                                                                          'src_mask': src_mask,
                                                                          }

    summary_corpus = []
    article_corpus = []
    candidate_sens = []
    candidate_combs = []
    labels = []
    if data_name == 'cnn':
        data = load_dataset('cnn_dailymail', '3.0.0')
        if split is None:
            splits = ['train']
        else:
            splits = split
        for splt in splits:
            dataset = data[splt]
            for model_data in tqdm(dataset):
                article_corpus.append(model_data['article'])
                summary_corpus.append(model_data['highlights'])
        # print(cnn_data)
    if data_name == 'cnn_ext':
        data = load_dataset('json', data_files={'train': ['./datasets/cnndm/train.jsonl'],
                                                'validation': ['./datasets/cnndm/val.jsonl'],
                                                'test': ['./datasets/cnndm/test.jsonl']})
    elif data_name in ['pubmed', 'xsum', 'reddit', 'wikihow', 'multinews']:
        data = load_dataset('json', data_files={'train': ['./datasets/others/train_{}.jsonl'.format(data_name)],
                                                'test': ['./datasets/others/test_{}.jsonl'.format(data_name)]}
                            )
    if split is None:
        splits = ['train']
    else:
        splits = split
    for splt in splits:
        dataset = data[splt]
        for model_data in tqdm(dataset):
            label = [i for i in range(len(model_data['label'])) if model_data['label'][i] == 1]
            if len(label) == 0:
                continue
            labels.append(label)
            article_corpus.append(model_data['text'])
            summary_corpus.append(model_data['summary'])
            candidate_sens.append(model_data['ext_idx'])
            candidate_combs.append(model_data['indices'])

    dtst = get_dataset(article_corpus, summary_corpus, labels, candidate_sens, candidate_combs)
    ds = Summarization_Dataset(dtst, image_size, sent_encoder_type=sent_encoder_type, summary_type=summary_type, summary_level=summary_level)
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collect_fn
    )

    return dataloader
