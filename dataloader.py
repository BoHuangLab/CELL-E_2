import os
import numpy as np
from PIL import Image, ImageSequence
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


def simple_conversion(seq):
    """Create 26-dim embedding"""
    chars = [
        "-",
        "M",
        "R",
        "H",
        "K",
        "D",
        "E",
        "S",
        "T",
        "N",
        "Q",
        "C",
        "U",
        "G",
        "P",
        "A",
        "V",
        "I",
        "F",
        "Y",
        "W",
        "L",
        "O",
        "X",
        "Z",
        "B",
        "J",
    ]

    nums = range(len(chars))

    seqs_x = np.zeros(len(seq))

    for idx, char in enumerate(seq):

        lui = chars.index(char)

        seqs_x[idx] = nums[lui]

    return torch.tensor([seqs_x]).long()


def replace_outliers(image, percentile=0.0001):

    lower_bound, upper_bound = torch.quantile(image, percentile), torch.quantile(
        image, 1 - percentile
    )
    mask = (image <= upper_bound) & (image >= lower_bound)

    valid_pixels = image[mask]

    image[~mask] = torch.clip(image[~mask], min(valid_pixels), max(valid_pixels))

    return image


class CellLoader(Dataset):
    """imports mined opencell images with protein sequence"""

    def __init__(
        self,
        data_csv=None,
        dataset=None,
        split_key=None,
        resize=600,
        crop_size=600,
        crop_method="random",
        sequence_mode="simple",
        vocab="bert",
        threshold="median",
        text_seq_len=0,
        pad_mode="random",
    ):
        self.data_csv = data_csv
        self.dataset = dataset
        self.image_folders = []
        self.crop_method = crop_method
        self.resize = resize
        self.crop_size = crop_size
        self.sequence_mode = sequence_mode
        self.threshold = threshold
        self.text_seq_len = int(text_seq_len)
        self.vocab = vocab
        self.pad_mode = pad_mode

        if self.sequence_mode == "embedding" or self.sequence_mode == "onehot":


            if self.vocab == "esm1b" or self.vocab == "esm2":
                from esm import Alphabet

                self.tokenizer = Alphabet.from_architecture(
                    "ESM-1b"
                ).get_batch_converter()
                self.text_seq_len += 2

        if data_csv:

            data = pd.read_csv(data_csv)

            self.parent_path = os.path.dirname(data_csv).split(data_csv)[0]

            if split_key == "train":
                self.data = data[data["split"] == "train"]
            elif split_key == "val":
                self.data = data[data["split"] == "val"]
            else:
                self.data = data

            self.data = self.data.reset_index(drop=True)
            
                

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        idx,
        get_sequence=True,
        get_images=True,
    ):
        if get_sequence and self.text_seq_len > 0:

            protein_vector = self.get_protein_vector(idx)

        else:
            protein_vector = torch.zeros((1, 1))

        if get_images:

            nucleus, target, threshold = self.get_images(idx, self.dataset)
        else:
            nucleus, target, threshold = torch.zeros((3, 1))

        data_dict = {
            "nucleus": nucleus.float(),
            "target": target.float(),
            "threshold": threshold.float(),
            "sequence": protein_vector.long(),
        }

        return data_dict

    def get_protein_vector(self, idx):

        if "protein_sequence" not in self.data.columns:

            metadata = self.retrieve_metadata(idx)
            protein_sequence = metadata["sequence"]
        else:
            protein_sequence = self.data.iloc[idx]["protein_sequence"]

        protein_vector = self.tokenize_sequence(protein_sequence)

        return protein_vector

    def get_images(self, idx, dataset):

        if dataset == "HPA":

            nucleus = Image.open(
                os.path.join(
                    self.parent_path, self.data.iloc[idx]["nucleus_image_path"]
                )
            )

            target = Image.open(
                os.path.join(self.parent_path, self.data.iloc[idx]["target_image_path"])
            )

            nucleus = TF.to_tensor(nucleus)[0]
            target = TF.to_tensor(target)[0]

            image = torch.stack([nucleus, target], axis=0)

            normalize = (0.0655, 0.0650), (0.1732, 0.1208)

        elif dataset == "OpenCell":
            image = Image.open(
                os.path.join(self.parent_path, self.data.iloc[idx]["image_path"])
            )
            nucleus, target = [page.copy() for page in ImageSequence.Iterator(image)]

            nucleus = replace_outliers(torch.divide(TF.to_tensor(nucleus), 65536))[0]
            target = replace_outliers(torch.divide(TF.to_tensor(target), 65536))[0]

            image = torch.stack([nucleus, target], axis=0)

            normalize = (
                (0.0272, 0.0244),
                (0.0486, 0.0671),
            )

        # # from https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914

        t_forms = [transforms.Resize(self.resize, antialias=None)]

        if self.crop_method == "random":

            t_forms.append(transforms.RandomCrop(self.crop_size))
            t_forms.append(transforms.RandomHorizontalFlip(p=0.5))
            t_forms.append(transforms.RandomVerticalFlip(p=0.5))

        elif self.crop_method == "center":

            t_forms.append(transforms.CenterCrop(self.crop_size))

        t_forms.append(transforms.Normalize(normalize[0], normalize[1]))

        image = transforms.Compose(t_forms)(image)

        nucleus, target = image

        nucleus /= torch.abs(nucleus).max()
        target -= target.min()
        target /= target.max()

        nucleus = nucleus.unsqueeze(0)
        target = target.unsqueeze(0)

        threshold = target

        if self.threshold == "mean":

            threshold = 1.0 * (threshold > (torch.mean(threshold)))

        elif self.threshold == "median":

            threshold = 1.0 * (threshold > (torch.median(threshold)))

        elif self.threshold == "1090_IQR":

            p10 = torch.quantile(threshold, 0.1, None)
            p90 = torch.quantile(threshold, 0.9, None)
            threshold = torch.clip(threshold, p10, p90)

        nucleus = torch.nan_to_num(nucleus, 0.0, 1.0, 0.0)
        target = torch.nan_to_num(target, 0.0, 1.0, 0.0)
        threshold = torch.nan_to_num(threshold, 0.0, 1.0, 0.0)

        return nucleus, target, threshold

    def retrieve_metadata(self, idx):
        with open(
            os.path.join(self.parent_path, self.data.iloc[idx]["metadata_path"])
        ) as f:
            metadata = json.load(f)

        return metadata

    def tokenize_sequence(self, protein_sequence):

        pad_token = 0

        if self.sequence_mode == "simple":
            protein_vector = simple_conversion(protein_sequence)

        elif self.sequence_mode == "center":
            protein_sequence = protein_sequence.center(self.text_seq_length, "-")
            protein_vector = simple_conversion(protein_sequence)

        elif self.sequence_mode == "alternating":
            protein_sequence = protein_sequence.center(self.text_seq_length, "-")
            protein_sequence = protein_sequence[::18]
            protein_sequence = protein_sequence.center(
                int(self.text_seq_length / 18) + 1, "-"
            )
            protein_vector = simple_conversion(protein_sequence)


        elif self.sequence_mode == "embedding":

            if self.vocab == "esm1b" or self.vocab == "esm2":
                pad_token = 1
                protein_vector = self.tokenizer([("", protein_sequence)])[-1]

        if protein_vector.shape[-1] < self.text_seq_len:

            diff = self.text_seq_len - protein_vector.shape[-1]

            if self.pad_mode == "end":
                protein_vector = torch.nn.functional.pad(
                    protein_vector, (0, diff), "constant", pad_token
                )
            elif self.pad_mode == "random":
                split = diff - np.random.randint(0, diff + 1)

                protein_vector = torch.cat(
                    [torch.ones(1, split) * 0, protein_vector], dim=1
                )

                protein_vector = torch.nn.functional.pad(
                    protein_vector, (0, diff - split), "constant", pad_token
                )

        elif protein_vector.shape[-1] > self.text_seq_len:
            start_int = np.random.randint(
                0, protein_vector.shape[-1] - self.text_seq_len
            )

            protein_vector = protein_vector[
                :, start_int : start_int + self.text_seq_len
            ]

        return protein_vector.long()
