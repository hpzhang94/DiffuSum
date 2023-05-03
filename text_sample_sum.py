"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from functools import partial

import numpy as np
import torch
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.nn.functional as F
from tqdm import tqdm

from imp_diff.improved_diffusion import dist_util, logger
from imp_diff.improved_diffusion.nn import mean_flat
from imp_diff.improved_diffusion.script_util import create_model_and_diffusion, args_to_dict, \
    model_and_diffusion_defaults, add_dict_to_argparser
from imp_diff.improved_diffusion.summarization_datasets_exp import load_data_summarization
from imp_diff.improved_diffusion.test_util import get_weights, compute_logp
from transformers import set_seed

def main():
    set_seed(101)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.sigma_small = True

    # args.diffusion_steps = 200 #500  # DEBUG
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print("Loading Model from: {}".format(args.model_path))
    model.load_state_dict(
        torch.load(args.model_path)
    )

    model.to(dist_util.dev())

    model.eval()  # DEBUG

    print("Model Type: {}".format(type(model))) # Transformer 3 Model

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    batch_size = 128
    data_test = load_data_summarization(batch_size, 8, training_args['roc_train'], split=['test'],
                                   sent_encoder_type='sbert', summary_type='oracle',
                                   summary_level='sen', shuffle=False)

    logger.log("sampling...")
    all_images = []
    all_article_sen_embs = []
    all_article_sen_masks = []
    all_article_sens = []
    all_summary_sens = []
    all_cand_sens = []
    all_cand_combs = []
    with torch.no_grad():
        for i, (batch, model_kwargs) in tqdm(enumerate(data_test)):
            summary_sens = model_kwargs['summary_sens']
            article_sens = model_kwargs['article_sens']
            all_article_sens += article_sens
            all_summary_sens += summary_sens
            article_sen_embs = model_kwargs['article_sen_embeds']
            article_sen_masks = model_kwargs['article_sen_masks']
            all_article_sen_masks.append(article_sen_masks)
            all_article_sen_embs.append(article_sen_embs)
            all_cand_sens += (model_kwargs['cand_sens'])
            all_cand_combs += (model_kwargs['cand_combs'])

            sample_fn = (
                diffusion.p_sample_loop
            )

            sample_shape = (batch_size, args.image_size ** 2, args.in_channel)
            sample = sample_fn(
                model,
                sample_shape,
                clip_denoised=args.clip_denoised,
                denoised_fn= None ,
                model_kwargs=model_kwargs,
                top_p =-1,
            )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        arr = np.concatenate(all_images, axis=0)
        print(arr.shape, 'full shape')
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
        out_path = os.path.join(args.out_dir, f"{model_base_name}.arr.npy")
        print("Saved Arr to :{}".format(out_path))

        if diffusion.training_mode.startswith('e2e'):
            preds = []
            summaries = []
            x_t = th.tensor(arr).cuda()
            all_article_cand_embs = torch.zeros_like(x_t[:, :article_sen_embs.shape[1], :]).cuda()
            all_artice_cand_mask = torch.zeros((all_article_cand_embs.shape[0], 1, all_article_cand_embs.shape[1])).cuda()
            for bid, cands in enumerate(all_cand_sens):
                for j, cand_idx in enumerate(cands):
                    all_article_cand_embs[bid, cand_idx, :] = x_t[bid, cand_idx, :]
                    all_artice_cand_mask[bid, :, cand_idx] = 1

            x_sum = x_t[:, article_sen_embs.shape[1]:, :]

            if args.model_arch == 'conv-unet':
                reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
            else:
                reshaped_x_t = x_sum

            logits = th.softmax(torch.bmm(reshaped_x_t, x_t[:, :article_sen_embs.shape[1], :].permute(0, 2, 1)),
                                dim=-1).squeeze()  # bsz, seqlen, vocab
            logits *= all_artice_cand_mask.cuda()
            cands = th.topk(logits, k=5, dim=-1)

            sample = cands.indices
            for i, (indice, value) in enumerate(zip(cands.indices, cands.values)):
                end_idx = article_sen_embs.shape[1]

                pred_idxs =[]
                if training_args['roc_train'] == 'cnn_ext':
                    max_summary_sen_num = 3
                elif training_args['roc_train'] == 'pubmed':
                    max_summary_sen_num = 6
                elif training_args['roc_train'] == 'wikihow':
                    max_summary_sen_num = 4
                elif training_args['roc_train'] == 'multinews':
                    max_summary_sen_num = 9
                elif  training_args['roc_train'] == 'xsum':
                    max_summary_sen_num = 2

                for candis, cands_val in zip(indice, value):
                    for idx, val in zip(candis, cands_val):
                        if idx not in pred_idxs:
                            pred_idxs.append(int(idx))
                            break
                    if len(pred_idxs) == max_summary_sen_num:
                        break

                pred_idxs = sorted([idx for idx in pred_idxs if idx != end_idx])
                tokens = " ".join([all_article_sens[i][x] for x in pred_idxs if x < len(all_article_sens[i])])
                summary = " ".join(all_summary_sens[i])

                preds.append(tokens)
                summaries.append(summary)
            dict = {'preds': preds, 'summaries': summaries, 'cands': sample}

            model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
            out_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.npy")
            logger.log(f"saving to {out_path}")
            np.save(out_path, dict)
            logger.log(f"Finished saving to {out_path}")



def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,#10000,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        gradient_clipping=-1.0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='conv-unet',
                         roc_train='diffusion_lm/ROCstory',  # 'diffusion_lm/ROCstory/ROCstory17.csv',
                         wiki_train='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
                         e2e_train='e2e_data',
                         yelp_train='diffusion_lm/yelpnlg-resources/yelpnlg-corpus',
                         commonGen_train='diffusion_lm/common-gen/commongen_data',
                         emb_scale_factor=1.0, noise_level=0.0, cache_mode='no', use_bert_tokenizer='no',
                         padding_mode='block',
                         preprocessing_num_workers=1, top_p=-1.0)
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
