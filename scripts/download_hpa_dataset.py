import os
from multiprocessing.pool import Pool
import time
import pandas as pd
import requests
from tqdm import tqdm
import os
from PIL import Image
from io import BytesIO
import json

parent_path = "/home/emaad/CELL-E_2/data/HPA/"


def uniprot_request(url):

    try:
        return requests.get(url)
    except:
        time.sleep(30)
        return uniprot_request(url)


def retrieve_sequence(id):

    url = f"https://rest.uniprot.org/uniprotkb/{id}.fasta"

    result = uniprot_request(url)

    if result.ok:
        name = " ".join(result.text.split("OS")[0].split(" ")[1:-1])
        sequence = result.text.split("\n")[1:-1]
        sequence = "".join(sequence)

        return name, sequence
    else:
        time.sleep(30)
        retrieve_sequence(id)


def pull_images(url):

    _image = requests.get(url)

    if _image.ok:

        image = Image.open(BytesIO(_image.content))

        return image

    else:

        time.sleep(30)

        return pull_images(url)


def download_images(data, pid, start_idx, end_idx):
    for idx in tqdm(range(start_idx, end_idx), postfix=pid):
        ensg_id, name, locations, uniprot_id, cell_line, nucleus, target = data.iloc[
            [idx]
        ].values[0]

        prot_id = nucleus.split("/")[-2]

        split_nuc = nucleus.split("/")[-1].split("_")
        cell_id = split_nuc[0]
        im_id = split_nuc[2]
        top_folder = os.path.join(parent_path, prot_id)

        if not os.path.exists(top_folder):
            os.mkdir(top_folder)

        if not os.path.exists(os.path.join(top_folder, "metadata.json")):

            protein_name, sequence = retrieve_sequence(uniprot_id)

            metadata = {
                "ensg_id": ensg_id,
                "protein_name": protein_name,
                "target_name": name,
                "unitprot_id": uniprot_id,
                "locations": locations,
                "sequence": sequence,
            }

            with open(os.path.join(top_folder, "metadata.json"), "w") as fp:
                json.dump(metadata, fp)

        cell_folder = os.path.join(top_folder, cell_id)

        if not os.path.exists(cell_folder):
            os.mkdir(cell_folder)

        im_folder = os.path.join(cell_folder, im_id)

        if not os.path.exists(im_folder):
            os.mkdir(im_folder)

        pull_images(nucleus).split()[2].save(os.path.join(im_folder, "nucleus.jpg"))
        time.sleep(0.1)
        pull_images(target).split()[1].save(os.path.join(im_folder, "target.jpg"))


def run_proc(name, start_idx, end_idx):
    """Handle one mp process."""
    print(
        "Run child process %s (%s) start: %d end: %d"
        % (name, os.getpid(), start_idx, end_idx)
    )
    download_images(data, name, start_idx, end_idx)
    print("Run child process %s done" % (name))


def download_hpa(process_num, data):
    print("Parent process %s." % os.getpid())
    list_len = len(data)
    pool = Pool(process_num)
    for i in range(process_num):
        pool.apply_async(
            run_proc,
            args=(
                str(i),
                int(i * list_len / process_num),
                int((i + 1) * list_len / process_num),
            ),
        )
    print("Waiting for all subprocesses done...")
    pool.close()
    pool.join()
    print("All subprocesses done.")


if __name__ == "__main__":
    data = pd.read_csv(parent_path + "HPAdata.csv")

    download_hpa(5, data)
