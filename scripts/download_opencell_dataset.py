import os
from multiprocessing.pool import Pool
import time
import requests
from tqdm import tqdm
import os
import json


parent_path = "."


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


API_KEY = ""

LINES_API = "http://opencell.czbiohub.org/api/lines?publication_ready=true"

LINE_FOVS_API = "http://opencell.czbiohub.org/api/lines/{line_id}/fovs?fields=rois&onlyannotated=true"

ROI_API = "http://opencell.czbiohub.org/api/rois/{roi_id}/proj/"

headers = {
    "authorization": f"Basic {API_KEY}",
    "accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
    "referer": "http://opencell.czbiohub.org/gallery",
}


def download_images(lines, pid, start_idx, end_idx):
    for idx in tqdm(range(start_idx, end_idx), postfix=pid):
        line = lines[idx]
        if not os.path.exists(os.path.join(top_folder, "metadata.json")):
            print(line)
            metadata = line["metadata"]
            uniprot_id = line["uniprot_metadata"]["uniprot_id"]
            protein_name, sequence = retrieve_sequence(uniprot_id)
            prot_folder = metadata["target_name"] + "_" + metadata["ensg_id"]
            top_folder = os.path.join(parent_path, prot_folder)

            try:
                metadata.update({"protein_name": protein_name})
                metadata.update({"uniprot_id": uniprot_id})
                metadata.update({"sequence": sequence})

                with open(os.path.join(top_folder, "metadata.json"), "w") as fp:
                    json.dump(metadata, fp)
            except:
                print(top_folder)


def run_proc(lines, name, start_idx, end_idx):
    """Handle one mp process."""
    print(
        "Run child process %s (%s) start: %d end: %d"
        % (name, os.getpid(), start_idx, end_idx)
    )
    download_images(lines, name, start_idx, end_idx)
    print("Run child process %s done" % (name))


def download_opencell(process_num, lines):
    """Download HPAv21 images with a multiprocessing pool."""
    print("Parent process %s." % os.getpid())
    list_len = len(lines)
    pool = Pool(process_num)
    for i in range(process_num):
        pool.apply_async(
            run_proc,
            args=(
                lines,
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
    response = requests.get(LINES_API, headers=headers)
    lines = response.json()

    download_opencell(1, lines)
