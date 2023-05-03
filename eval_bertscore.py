from evaluate import load
import numpy as np

def eval_bertscore(preds, summaries):
    bertscore = load("bertscore")
    scores = bertscore.compute(predictions=preds, references=summaries, lang="en")
    res_dict = {
        "precision": np.mean(scores['precision']),
        "recall": np.mean(scores['recall']),
        "f1": np.mean(scores['f1']),
    }
    return res_dict
