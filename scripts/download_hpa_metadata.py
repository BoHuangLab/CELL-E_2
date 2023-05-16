import os
import xml.etree.cElementTree as ET
from multiprocessing.pool import Pool
import time
import pandas as pd
import requests
from tqdm import tqdm


def get_data(gene):
    try:
        tree = ET.fromstring(
            requests.get(f"https://www.proteinatlas.org/{gene}.xml").text
        )
        time.sleep(0.1)
    except:
        return None
    df = pd.DataFrame(columns=["cell_line", "nucleus", "target"])

    xml = tree.find("entry")
    locations = xml.find("predictedLocation").text
    for xref in xml.find("identifier").findall("xref"):
        if "uniprot" in xref.get("db").lower():
            uniprot_id = xref.get("id")

    try:
        for antibody in xml.findall("antibody"):
            if antibody.find("cellExpression"):
                for data in (
                    antibody.find("cellExpression").find("subAssay").findall("data")
                ):
                    for url in data.findall("*//imageUrl"):
                        base_url = url.text.split("blue")[0]
                        nucleus = base_url + "blue.jpg"
                        target = base_url + "green.jpg"
                        df = df.append(
                            {
                                "cell_line": data.find("cellLine").text,
                                "nucleus": nucleus,
                                "target": target,
                            },
                            ignore_index=True,
                        )
        df["locations"] = locations
        df["uniprot_id"] = uniprot_id

        return df
    except:
        return None


def download_images(subcell_data, pid, start_idx, end_idx):
    for idx in tqdm(range(start_idx, end_idx), postfix=pid):
        gene, name = subcell_data[["Gene", "Gene name"]].iloc[[idx]].values[0]
        df = get_data(gene)
        time.sleep(0.1)
        if df is not None:
            df["ensg_id"] = gene
            df["name"] = name
            df = df[
                [
                    "ensg_id",
                    "name",
                    "locations",
                    "uniprot_id",
                    "cell_line",
                    "nucleus",
                    "target",
                ]
            ]
            df.to_csv(
                "/home/emaad/CELL-E_2/data/HPAdata.csv",
                mode="a",
                header=False,
                index=False,
            )
        else:
            text_file = open("/home/emaad/CELL-E_2/data/failures.txt", "a")
            text_file.write(f"{gene}\n")
            text_file.close()


def run_proc(name, start_idx, end_idx):
    """Handle one mp process."""
    print(
        "Run child process %s (%s) start: %d end: %d"
        % (name, os.getpid(), start_idx, end_idx)
    )
    download_images(subcell_data, name, start_idx, end_idx)
    print("Run child process %s done" % (name))


def download_hpa_v18(process_num, subcell_data):
    """Download HPAv18 images with a multiprocessing pool."""
    print("Parent process %s." % os.getpid())
    list_len = len(subcell_data)
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
    subcell_data = pd.read_csv("./subcellular_location.tsv", sep="\t")
    count = 0
    current_list = pd.read_csv("/home/emaad/CELL-E_2/data/HPAdata.csv")
    missed_genes = list(
        set(pd.unique(subcell_data["Gene"])) - set(pd.unique(current_list["ensg_id"]))
    )
    filtered_data = subcell_data[subcell_data["Gene"].isin(missed_genes)]
    while len(filtered_data) > 0:
        print(
            f"Missed Genes: {len(missed_genes)} - { len(missed_genes) / len(pd.unique(subcell_data['Gene']))*100:.2f} %"
        )
        text_file = open("/home/emaad/CELL-E_2/data/failures.txt", "a")
        count += 1
        text_file.write(f"Round #{count}:\n")
        text_file.close()
        download_hpa_v18(5, filtered_data)
        current_list_2 = pd.read_csv("/home/emaad/CELL-E_2/data/HPAdata.csv")
        missed_genes = list(
            set(pd.unique(subcell_data["Gene"]))
            - set(pd.unique(current_list_2["ensg_id"]))
        )
        filtered_data = subcell_data[subcell_data["Gene"].isin(missed_genes)]
        filtered_data = filtered_data.sample(frac=1).reset_index(drop=True)
