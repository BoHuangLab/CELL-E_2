import os, sys

import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from celle_main import instantiate_from_config
from omegaconf import OmegaConf
import numpy as np
from dataloader import CellLoader
from tqdm import tqdm
from cellpose import models
import gc

seq_config_path = ""
seq_ckpt_path = ""
im_config_path = ""
im_ckpt_path = ""

batch_size = 9
total_seqs = 300
NLS_min_len = 5
NLS_max_len = 30
SEQUENCE = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
output_dir = ""


def de_novo_nls_search():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = OmegaConf.load(seq_config_path)
    config['model']['params']['condition_model_path'] = None
    config['model']['params']['vqgan_model_path'] = None
    if config["model"]["params"]["ckpt_path"] is None:
        config["model"]["params"]["ckpt_path"] = seq_ckpt_path
    seq_model = instantiate_from_config(config.model)
    seq_model = seq_model.to(device)
    seq_model = seq_model.eval()

    if not im_ckpt_path or im_ckpt_path == seq_ckpt_path:
        im_model = seq_model
    else:
        config = OmegaConf.load(im_config_path)
        config['model']['params']['condition_model_path'] = None
        config['model']['params']['vqgan_model_path'] = None
        if config["model"]["params"]["ckpt_path"] is None:
            config["model"]["params"]["ckpt_path"] = im_ckpt_path
        im_model = instantiate_from_config(config.model)
        im_model = im_model.to(device)
        im_model = im_model.eval()

    dataset = CellLoader(
        data_csv="data/OpenCell/train_test_split_HPA_50.csv",
        dataset="OpenCell",
        sequence_mode="embedding",
        vocab="esm2",
        split_key="val",
        crop_method="center",
        resize=600,
        crop_size=600,
        text_seq_len=1000,
        pad_mode="end",
        threshold="median",
    )

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    if "cuda" in device:
        gpu = True
    else:
        gpu = False

    nuc_segmentation = models.Cellpose(
        model_type="nuclei", gpu=gpu, device=torch.device(device)
    )

    target_sequences_df = pd.DataFrame(
        columns=["sequence", "length", "terminus", "nucleus_proportion", "text_confidence"]
    )

    # pick nucleus
    # use cellpose to generate nuclear mask

    for idx, batch in enumerate(loader):
        nucleus_og = batch["nucleus"]
        nucleus = nucleus_og.to(device)

        if idx == 0:
            break

    masks, _, _, _ = nuc_segmentation.eval(
        [img.numpy() for img in nucleus_og],
        diameter=80,
        flow_threshold=None,
        channels=[0, 0],
    )

    masks = torch.from_numpy(np.array(masks, dtype=np.float16)).to(device)
    masks = 1.0 * (masks > 0)

    nucleus = torch.cat([nucleus, masks.unsqueeze(1)], dim=1)


    # Need to iterate through time steps and sample vs sample_text


    for NLS_len in tqdm(range(NLS_min_len, NLS_max_len + 1)):
        sequence_n = f'{sequence[:1]}{NLS_len*"<mask>"}{sequence[1:]}'
        sequence_c = f'{sequence}{NLS_len*"<mask>"}'
        sequence_n = dataset.tokenize_sequence(sequence_n).to(device)
        sequence_c = dataset.tokenize_sequence(sequence_c).to(device)

        count = 0

        try:
            while count < total_seqs:
                # pass sequence through sample function

                with torch.no_grad():
                    nucleus = nucleus[torch.randperm(nucleus.shape[0])].to(device)
                    nucleus = transforms.Compose(
                        [
                            transforms.RandomCrop(256),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                        ]
                    )(nucleus)

                    text_nums, text, text_logits = seq_model.celle.sample_text(
                        text=sequence_n.repeat(batch_size, 1),
                        condition=nucleus[:, :1],
                        image=nucleus[:, 1:],
                        n_unmask=1,
                        temperature=1,
                        filter_thres=0.95,
                        force_aas=True,
                    )

                    text_logits = text_logits[:, :, 4:-9]

                    text_confidence = torch.mean(
                        torch.max(
                            text_logits[:, 1 : NLS_len + 1]
                            / torch.norm(text_logits[:, 1 : NLS_len + 1, :32], dim=-1)
                            .unsqueeze(-1)
                            .repeat(1, 1, text_logits.shape[-1]),
                            dim=-1,
                        )[0],
                        dim=-1,
                    )

                    torch.cuda.empty_cache()
                    gc.collect()

                    # predict image and make sure its Nuclear (save percentage)
                    _, _, _, image, heatmap = im_model.celle.sample(
                        text=text_nums,
                        condition=nucleus[:, :1],
                        timesteps=1,
                        temperature=1,
                        filter_thres=0.95,
                    )

                    nucleus_proportion = torch.sum(
                        image * nucleus[:, 1:], dim=(1, 2, 3)
                    ) / torch.sum(image, dim=(1, 2, 3))

                    # save logits as "self-confidence" number

                    image_confidence = torch.sum(
                        heatmap * nucleus[:, 1:], dim=(1, 2, 3)
                    ) / torch.sum(heatmap, dim=(1, 2, 3))

                    start_token = (text_nums == 0).nonzero(as_tuple=True)[1][0]

                    for idx, item in enumerate(text_nums):
                        NLS = seq_model.celle.detokenize(
                            item[start_token + 2 : start_token + 2 + NLS_len].unsqueeze(0)
                        )

                        metrics = {
                            "sequence": NLS[0],
                            "length": NLS_len,
                            "terminus": "N",
                            "nucleus_proportion": nucleus_proportion[idx].item(),
                            "image_confidence": image_confidence[idx].item(),
                            "text_confidence": text_confidence[idx].item(),
                        }

                        for key, value in metrics.items():
                            if torch.is_tensor(value):
                                value = value.item()
                            metrics[key] = [value]

                        metrics = pd.DataFrame.from_dict(metrics)
                        target_sequences_df = pd.concat(
                            [target_sequences_df, metrics], ignore_index=True
                        )

                    # pass sequence through sample function

                with torch.no_grad():
                    text_nums, text, text_logits = seq_model.celle.sample_text(
                        text=sequence_c.repeat(batch_size, 1),
                        condition=nucleus[:, :1],
                        image=nucleus[:, 1:],
                        n_unmask=1,
                        temperature=1,
                        filter_thres=0.95,
                        force_aas=True,
                    )

                    torch.cuda.empty_cache()
                    gc.collect()

                    text_logits = text_logits[:, :, 4:-9]

                    text_confidence = torch.mean(
                        torch.max(
                            text_logits[:, len(sequence) + 1 : len(sequence) + NLS_len + 1]
                            / torch.norm(
                                text_logits[:, len(sequence) + 1 : len(sequence) + NLS_len + 1],
                                dim=-1,
                            )
                            .unsqueeze(-1)
                            .repeat(1, 1, text_logits.shape[-1]),
                            dim=-1,
                        )[0],
                        dim=-1,
                    )


                    # predict image and make sure its Nuclear (save percentage)
                    _, _, _, image, heatmap = im_model.celle.sample(
                        text=text_nums,
                        condition=nucleus[:, :1],
                        timesteps=1,
                        temperature=1,
                        filter_thres=0.95,
                    )

                    nucleus_proportion = torch.sum(
                        image * nucleus[:, 1:], dim=(1, 2, 3)
                    ) / torch.sum(image, dim=(1, 2, 3))

                    # save logits as "self-confidence" number

                    image_confidence = torch.sum(
                        heatmap * nucleus[:, 1:], dim=(1, 2, 3)
                    ) / torch.sum(heatmap, dim=(1, 2, 3))


                    end_token = (text_nums == 2).nonzero(as_tuple=True)[1][0]

                    for idx, item in enumerate(text_nums):
                        NLS = seq_model.celle.detokenize(
                            item[end_token - NLS_len : end_token].unsqueeze(0)
                        )

                        metrics = {
                            "sequence": NLS[0],
                            "length": NLS_len,
                            "terminus": "C",
                            "nucleus_proportion": nucleus_proportion[idx].item(),
                            "image_confidence": image_confidence[idx].item(),
                            "text_confidence": text_confidence[idx].item(),
                        }

                        for key, value in metrics.items():
                            if torch.is_tensor(value):
                                value = value.item()
                            metrics[key] = [value]

                        metrics = pd.DataFrame.from_dict(metrics)
                        target_sequences_df = pd.concat(
                            [target_sequences_df, metrics], ignore_index=True
                        )
                count += batch_size

        except:
            print(f"FAILED AT {NLS_len}")

    if os.path.exists(output_dir):
        output_csv = pd.read_csv(output_dir)
        target_sequences_df = pd.concat(
            [output_csv, target_sequences_df], ignore_index=True
        )

    target_sequences_df.to_csv(output_dir)