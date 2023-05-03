import json
import os
import time
from multiprocessing import Pool

import shutil
import nltk
import logging

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
from pyrouge import Rouge155



def process(data, tmp='tmp_pyrouge'):
    # tmp='/tmp/tmp_pyrouge'
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = str(time.time()).replace('.', '')
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    tmp_dir = tmp + "/{}{}".format(current_time, pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")

    def write(url, s):
        with open(url, 'w', encoding='utf-8') as f:
            f.write(s)

    for i in range(cnt):
        if len(references[i]) < 1:
            continue

        write(tmp_dir + "/candidate/cand.{}.txt".format(i), candidates[i])
        write(tmp_dir + "/reference/ref.{}.txt".format(i), references[i])

    r = Rouge155()
    r.log.setLevel(logging.WARN)
    r.model_dir = tmp_dir + "/reference/"
    r.system_dir = tmp_dir + "/candidate/"
    r.model_filename_pattern = 'ref.#ID#.txt'
    r.system_filename_pattern = r'cand.(\d+).txt'
    rouge_results = r.convert_and_evaluate()

    results_dict = r.output_to_dict(rouge_results)

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return results_dict


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def eval_rouge(cand, ref, num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]

    print("system output length: ",len(candidates))
    print("reference output length: ",len(references))
    assert len(candidates) == len(references)
    candidates_chunks = list(chunks(candidates, int(len(candidates) / num_processes)))
    references_chunks = list(chunks(references, int(len(references) / num_processes)))
    n_pool = len(candidates_chunks)
    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i], references_chunks[i], i))
    pool = Pool(n_pool)
    results = pool.map(process, arg_lst)
    final_results = {}
    for i, r in enumerate(results):
        for k in r:
            if (k not in final_results):
                final_results[k] = r[k] * len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])
    for k in final_results:
        final_results[k] = final_results[k] / len(candidates)
    return final_results


# preds are lists consists of summaries.
# each summary is a string: sentence1, \n , sentence2, \n
def pyrouge_score(preds, labels, num_processes=1, use_stemmer=True):
    def split_sent(s):
        return '\n'.join(nltk.sent_tokenize(s))
    if use_stemmer:
        stemmer = PorterStemmer()
        for i, pred in enumerate(preds):
            words = [stemmer.stem(word) for word in word_tokenize(pred)]
            # print(words)
            preds[i] = " ".join(words)
        # preds = [" ".join(stemmer.stem(word)) for pred in preds for word in word_tokenize(pred)]
        for i, label in enumerate(labels):
            words = [stemmer.stem(word) for word in word_tokenize(label)]
            labels[i] = " ".join(words)
    # print(preds)

    preds = list(map(split_sent, preds))
    labels = list(map(split_sent, labels))

    results = eval_rouge(cand=preds, ref=labels, num_processes=num_processes)
    res_dict = {'rouge1': results['rouge_1_f_score'] * 100, 'rouge2': results['rouge_2_f_score'] * 100,
                'rougeLsum': results['rouge_l_f_score'] * 100}
    return res_dict

