import argparse
import torch
import os
os.chdir('..')
from dataloader import CellLoader
from matplotlib import pyplot as plt
from celle_main import instantiate_from_config
from omegaconf import OmegaConf
from celle.utils import process_image

def run_model(mode, sequence,
            nucleus_image_path,
            protein_image_path,
            model_ckpt_path,
            model_config_path,
            device):
    if mode == "image":
        run_image_prediction(
            sequence,
            nucleus_image_path,
            protein_image_path,
            model_ckpt_path,
            model_config_path,
            device
        )
    elif mode == "sequence":
        run_sequence_prediction(
            sequence,
            nucleus_image_path,
            protein_image_path,
            model_ckpt_path,
            model_config_path,
            device
        )

def run_sequence_prediction(
    sequence_input,
    nucleus_image_path,
    protein_image_path,
    model_ckpt_path,
    model_config_path,
    device
):
    """
    Run Celle model with provided inputs and display results.

    :param sequence: Path to sequence file
    :param nucleus_image_path: Path to nucleus image
    :param protein_image_path: Path to protein image (optional)
    :param model_ckpt_path: Path to model checkpoint
    :param model_config_path: Path to model config
    """
    
    # Instantiate dataset object
    dataset = CellLoader(
        sequence_mode="embedding",
        vocab="esm2",
        split_key="val",
        crop_method="center",
        resize=600,
        crop_size=256,
        text_seq_len=1000,
        pad_mode="end",
        threshold="median",
    )

    # Check if sequence is provided and valid
    if len(sequence_input) == 0:
        raise ValueError("Sequence must be provided.")

    if "<mask>" not in sequence_input:
        print("Warning: Sequence does not contain any masked positions to predict.")

    # Convert SEQUENCE to sequence using dataset.tokenize_sequence()
    sequence = dataset.tokenize_sequence(sequence_input)

    # Check if nucleus image path is provided and valid
    if not os.path.exists(nucleus_image_path):
        # Use default nucleus image from dataset and print warning
        nucleus_image_path = 'images/nucleus.jpg'
        print(
            "Warning: No nucleus image provided. Using default nucleus image from dataset."
        )
    else:
        # Load nucleus image from provided path
        nucleus_image = process_image(nucleus_image_path)
        
    # Check if protein image path is provided and valid
    if not os.path.exists(protein_image_path):
        # Use default nucleus image from dataset and print warning
        protein_image_path = 'images/protein.jpg'
        print(
            "Warning: No nucleus image provided. Using default protein image from dataset."
        )
    else:
        # Load protein image from provided path
        protein_image = process_image(protein_image_path)
        protein_image = (protein_image > torch.median(protein_image,dim=0))*1.0

    # Load model config and set ckpt_path if not provided in config
    config = OmegaConf.load(model_config_path)
    if config["model"]["params"]["ckpt_path"] is None:
        config["model"]["params"]["ckpt_path"] = model_ckpt_path

    # Set condition_model_path and vqgan_model_path to None
    config["model"]["params"]["condition_model_path"] = None
    config["model"]["params"]["vqgan_model_path"] = None

    # Instantiate model from config and move to device
    model = instantiate_from_config(config).to(device)

    # Sample from model using provided sequence and nucleus image
    _, predicted_sequence, _ = model.celle.sample_text(
        text=sequence,
        condition=nucleus_image,
        image=protein_image,
        force_aas=True,
        timesteps=1,
        temperature=1,
        progress=True,
    )

    formatted_predicted_sequence = ""

    for i in range(min(len(predicted_sequence), len(sequence))):
        if predicted_sequence[i] != sequence[i]:
            formatted_predicted_sequence += f"**{predicted_sequence[i]}**"
        else:
            formatted_predicted_sequence += predicted_sequence[i]

    if len(predicted_sequence) > len(sequence):
        formatted_predicted_sequence += f"**{predicted_sequence[len(sequence):]}**"

    print("predicted_sequence:", formatted_predicted_sequence)


def run_image_prediction(
    sequence_input,
    nucleus_image_path,
    protein_image_path,
    model_ckpt_path,
    model_config_path,
    device
):
    """
    Run Celle model with provided inputs and display results.

    :param sequence: Path to sequence file
    :param nucleus_image_path: Path to nucleus image
    :param protein_image_path: Path to protein image (optional)
    :param model_ckpt_path: Path to model checkpoint
    :param model_config_path: Path to model config
    """
    # Instantiate dataset object
    dataset = CellLoader(
        sequence_mode="embedding",
        vocab="esm2",
        split_key="val",
        crop_method="center",
        resize=600,
        crop_size=256,
        text_seq_len=1000,
        pad_mode="end",
        threshold="median",
    )

    # Check if sequence is provided and valid
    if len(sequence_input) == 0:
        sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        # Use default sequence for GFP and print warning
        print("Warning: No sequence provided. Using default sequence for GFP.")

    # Convert SEQUENCE to sequence using dataset.tokenize_sequence()
    sequence = dataset.tokenize_sequence(sequence_input)

    # Check if nucleus image path is provided and valid
    if not os.path.exists(nucleus_image_path):
        # Use default nucleus image from dataset and print warning
        nucleus_image = dataset[0]["nucleus"]
        print(
            "Warning: No nucleus image provided. Using default nucleus image from dataset."
        )
    else:
        # Load nucleus image from provided path
        nucleus_image = process_image(nucleus_image_path)

    # Load model config and set ckpt_path if not provided in config
    config = OmegaConf.load(model_config_path)
    if config["model"]["params"]["ckpt_path"] is None:
        config["model"]["params"]["ckpt_path"] = model_ckpt_path

    # Set condition_model_path and vqgan_model_path to None
    config["model"]["params"]["condition_model_path"] = None
    config["model"]["params"]["vqgan_model_path"] = None

    # Instantiate model from config and move to device
    model = instantiate_from_config(config).to(device)

    # Sample from model using provided sequence and nucleus image
    _, _, _, predicted_threshold, predicted_heatmap = model.celle.sample(
        text=sequence,
        condition=nucleus_image,
        timesteps=1,
        temperature=1,
        progress=True,
    )

    # Move predicted_threshold and predicted_heatmap to CPU and select first element of batch
    predicted_threshold = predicted_threshold.cpu()[0, 0]
    predicted_heatmap = predicted_heatmap.cpu()[0, 0]

    # Create 3 or 4 panel plot depending on whether protein image path is provided
    fig, axs = plt.subplots(1, 3 if protein_image_path is None else 4)
    axs[0].imshow(nucleus_image)
    axs[0].set_title("Nucleus Input")
    axs[1].imshow(predicted_threshold)
    axs[1].set_title("Predicted Threshold")
    if protein_image_path is not None:
        protein_image = process_image(protein_image_path)
        axs[2].imshow(protein_image)
        axs[2].set_title("Protein Image")
    axs[-1].imshow(predicted_heatmap)
    axs[-1].set_title("Predicted Heatmap")
    plt.show()


if __name__ == "__main__":
    # Parse command line arguments for input parameters
    parser = argparse.ArgumentParser(
        description="Run Celle model with provided inputs."
    )
    parser.add_argument("--mode", type=str, default="", help="Sequence or Image")
    parser.add_argument(
        "--sequence", type=str, default="", help="Path to sequence file"
    )
    parser.add_argument(
        "--nucleus_image_path",
        type=str,
        default="images/nucleus.jpg",
        help="Path to nucleus image",
    )
    parser.add_argument(
        "--protein_image_path",
        type=str,
        default=None,
        help="Path to protein image (optional)",
    )
    parser.add_argument(
        "--model_ckpt_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model_config_path", type=str, required=True, help="Path to model config"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", required=True, help="device"
    )
    args = parser.parse_args()

    run_model(
        args.mode,
        args.sequence,
        args.nucleus_image_path,
        args.protein_image_path,
        args.model_ckpt_path,
        args.model_config_path,
        args.device
    )
