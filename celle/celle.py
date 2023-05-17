# Import necessary packages and modules
from math import floor, ceil
import torch
from torch import nn
import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange
from celle.utils import (
    exists,
    always,
    eval_decorator,
    gumbel_sample,
    top_k,
    gamma_func,
    DivideMax,
)

# Import additional modules from within the codebase
from celle.transformer import Transformer


def generate_mask(gamma_func, batch_size, length, device):
    # Get the number of `True` values in the mask for each batch element
    num_true_values = floor(gamma_func(torch.rand(1)) * length)

    # Generate a random sample of indices to set to `True` in the mask
    # The number of indices in the sample is determined by `num_true_values`
    indices = (
        torch.rand((batch_size, length), device=device)
        .topk(num_true_values, dim=1)
        .indices
    )

    # Create a binary mask tensor with `True` values at the sampled indices
    mask = torch.zeros((batch_size, length), dtype=torch.bool, device=device)
    mask.scatter_(dim=1, index=indices, value=True)

    return mask


def match_batch_size(text, condition, image, batch_size):
    """
    This function ensures all inputs to the sample function have the same batch size.
    """
    if text.shape[0] != batch_size:
        text = text.repeat(batch_size, 1)

    if condition.shape[0] != batch_size:
        condition = condition.repeat(batch_size, 1)

    if image.shape[0] != batch_size:
        image = image.repeat(batch_size, 1)

    return text, condition, image


def calc_unmask_probs(timestep, timesteps, gamma_func):
    if timestep == 1 or timesteps == 1:
        unmask_prob = 1
    else:
        unmask_prob = 1 - gamma_func(timestep)
    return unmask_prob


def calculate_logits(
    input_tokens, input_mask, logits_function, filter_thres, temperature
):
    logits, _, _ = logits_function(input_tokens, input_mask, return_encoding=False)
    filtered_logits = top_k(logits, thres=filter_thres)
    sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

    return logits, sample


def unmask_tokens(
    input_tokens,
    input_mask,
    num_masked_tokens,
    logits,
    sample,
    timestep,
    timesteps,
    gamma,
    filter_func=None,
    pad_token=None,
    mask_token=None,
    force_aas=True,
):
    sample = sample.masked_fill(~input_mask.unsqueeze(-1), -torch.inf)
    if filter_func:
        sample = filter_func(
            input_tokens, sample, force_aas, pad_token=pad_token, mask_token=mask_token
        )
    selected_token_probs, selected_tokens = torch.max(sample, dim=-1)

    unmask_prob = calc_unmask_probs(timestep, timesteps, gamma)
    num_tokens_to_unmask = max(1, ceil(unmask_prob * num_masked_tokens))

    _, top_k_indices = torch.topk(selected_token_probs, num_tokens_to_unmask, dim=-1)

    sample_mask = torch.zeros(
        input_tokens.shape, dtype=torch.bool, device=input_tokens.device
    )
    sample_mask.scatter_(dim=1, index=top_k_indices, value=True)

    unmasked_tokens = torch.where(sample_mask, selected_tokens, input_tokens)
    full_logits = torch.where(
        sample_mask.unsqueeze(-1), logits, torch.zeros_like(logits)
    )
    return unmasked_tokens, full_logits


def suppress_invalid_text_tokens(
    text,
    logits,
    start_token=None,
    end_token=None,
    pad_token=None,
    mask_token=None,
    force_aas=False,
):
    # Find the indices of start_token and end_token in tensor text along axis=1
    idx_start = (text == start_token).nonzero(as_tuple=True)[1]
    idx_end = (text == end_token).nonzero(as_tuple=True)[1]

    # For every position other than the index corresponding to the start index, set the values on the start index of dimension=2 to -torch.inf
    if idx_start.nelement() != start_token:
        try:
            mask = idx_start.unsqueeze(1) != torch.arange(
                logits.size(1), device=text.device
            )
            indices = torch.where(mask)
            logits[indices[0], indices[1], start_token] = -torch.inf
        except:
            pass

    # else:
    #     idx_start = torch.zeros(text.size(0), dtype=torch.long)

    # Similarly, for every position other than the index corresponding to the end index, set the values on the end index of dimension=2 to -torch.inf
    if idx_end.nelement() != 0:
        try:
            mask = idx_end.unsqueeze(1) != torch.arange(
                logits.size(1), device=text.device
            )
            indices = torch.where(mask)
            logits[indices[0], indices[1], end_token] = -torch.inf
        except:
            pass

    # else:
    #     idx_end = torch.full((text.size(0),), text.size(1) - 1, dtype=torch.long)

    if pad_token:
        if idx_start.nelement() != 0 and idx_end.nelement() != 0:
            try:
                # For every position between the indices of start_token and end_token, set the values for 1st index of dimension=2 equal to -torch.inf. Any value outside of that range should be set to torch.inf.
                mask = (
                    torch.arange(logits.size(1), device=text.device)
                    >= idx_start.unsqueeze(1)
                ) & (
                    torch.arange(logits.size(1), device=text.device)
                    <= idx_end.unsqueeze(1)
                )

                indices = torch.where(mask)
                logits[indices[0], indices[1], pad_token] = -torch.inf

                indices = torch.where(~mask)
                logits[indices[0], indices[1], pad_token] = torch.inf

            except:
                pass

        elif idx_start.nelement() != 0:
            try:
                mask = torch.arange(
                    logits.size(1), device=text.device
                ) < idx_start.unsqueeze(1)
                logits[indices[0], indices[1], pad_token] = torch.inf
            except:
                pass

        elif idx_end.nelement() != 0:
            try:
                mask = torch.arange(
                    logits.size(1), device=text.device
                ) > idx_end.unsqueeze(1)
                logits[indices[0], indices[1], pad_token] = torch.inf
            except:
                pass

    if force_aas:
        if pad_token:
            logits[:, :, pad_token] = -torch.inf
        logits[:, :, 3] = -torch.inf
        logits[:, :, 29:] = -torch.inf

    if mask_token:
        logits[:, :, mask_token] = -torch.inf

    return logits


def detokenize_text(text_embedding, sequence):
    if text_embedding == "esm1b" or text_embedding == "esm2":
        from esm import Alphabet

        alphabet = (
            Alphabet.from_architecture("ESM-1b").get_batch_converter().alphabet.all_toks
        )
    else:
        assert NameError("Detokenization only available for ESM mdodels")

    output_seqs = []

    for batch in sequence:
        converted_seq = [alphabet[idx] for idx in batch]
        converted_seq = "".join(converted_seq)
        output_seqs.append(converted_seq)

    return output_seqs

class ImageEmbedding(nn.Module):
    def __init__(self, num_tokens, dim):
        super(ImageEmbedding, self).__init__()
        self.image_embedding = nn.Embedding(num_tokens, dim)
        
    def forward(self, image):
        return self.image_embedding(image)


class ModelExtender(nn.Module):
    def __init__(self, vocab, out_features, fixed_embedding=False):
        super(ModelExtender, self).__init__()

        # Initialize the model according to the given vocabulary
        self.vocab = vocab
        
        if vocab == "esm1b":
            from esm import pretrained

            self.model, _ = pretrained.esm1b_t33_650M_UR50S()
            self.in_features = 1280
        elif vocab == "esm2":
            from esm import pretrained

            if out_features == 320:
                self.model, _ = pretrained.esm2_t6_8M_UR50D()
            elif out_features == 480:
                self.model, _ = pretrained.esm2_t12_35M_UR50D()
            elif out_features == 640:
                self.model, _ = pretrained.esm2_t30_150M_UR50D()
            elif out_features == 1280:
                self.model, _ = pretrained.esm2_t33_650M_UR50D()
            elif out_features == 2560:
                self.model, _ = pretrained.esm2_t36_3B_UR50D()
            else:
                self.model, _ = pretrained.esm2_t33_650M_UR50D()
            self.in_features = self.model.embed_dim

        # Set the number of output features and initialize the scaling layer
        self.out_features = out_features
        if self.in_features != self.out_features:
            self.scale_layer = nn.Linear(self.in_features, self.out_features)
        else:
            self.scale_layer = nn.Identity()

        # Determine whether to freeze the model's parameters
        self.fixed_embedding = fixed_embedding
        if self.fixed_embedding:
            self.model = self.model.eval()

    def forward(self, x, **kwargs):
        # If the model's parameters are fixed, use torch.no_grad()
        if self.fixed_embedding:
            with torch.no_grad():
                if self.vocab == "esm1b" or self.vocab == "esm2":
                    # Reduce sequence length dimension, get top layer representation tensor
                    x = self.model(x.squeeze(1), repr_layers=[self.model.num_layers])[
                        "representations"
                    ][self.model.num_layers]
                    # Tensor shape: (batch_size, hidden_size)
                else:
                    # Get top layer representation tensor
                    x = self.model(x, **kwargs)[0]
                    # Tensor shape: (batch_size, sequence_length, hidden_size)
        else:
            if self.vocab == "esm1b" or self.vocab == "esm2":
                # Reduce sequence length dimension, get top layer representation tensor
                x = self.model(x.squeeze(1), repr_layers=[self.model.num_layers])[
                    "representations"
                ][self.model.num_layers]
                # Tensor shape: (batch_size, hidden_size)
            else:
                # Get top layer representation tensor
                x = self.model(x, **kwargs)[0]
                # Tensor shape: (batch_size, sequence_length, hidden_size)

        # Scale the representation tensor if necessary
        if self.out_features != self.in_features:
            x = self.scale_layer(x)
            # Tensor shape: (batch_size, out_features)

        return x

class CELLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,  # The VAE model used to encode/decode images
        condition_vae=None,  # An optional VAE model used to condition the image generation
        num_images=2,  # Number of images to generate
        num_text_tokens=30,  # Number of tokens in the text vocabulary
        text_seq_len=1000,  # Maximum length of input text sequence
        depth=16,  # Number of layers in the transformer model
        heads=16,  # Number of attention heads
        dim_head=64,  # Dimensionality of each attention head
        attn_dropout=0.1,  # Dropout rate for attention weights
        ff_dropout=0.1,  # Dropout rate for feedforward layers
        attn_types=None,  # Types of attention to use in the transformer
        causal=False,  # Whether to use causal attention
        loss_cond_weight=1,  # Weight of conditioning loss
        loss_img_weight=1,  # Weight of image generation loss
        stable=False,  # Whether to use divide-by-max normalization in the transformer
        rotary_emb=True,  # Whether to use rotary positional embeddings
        text_embedding="esm2",  # Text embedding to use (esm1b, esm2)
        fixed_embedding=True,  # Whether to fix the text embedding or learn it
        sampling_mode="cosine",  # Sampling mode for the VAE
        linear_project=False,  # Whether to project embeddings linearly
        **kwargs,
    ):
        super().__init__()

        # Set the stable flag
        self.stable = stable

        # If the stable flag is set, initialize the DivideMax layer for normalization
        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        ### Initializing text parameters ###

        # Initialize the text and fixed embeddings
        self.text_embedding = text_embedding
        self.fixed_embedding = fixed_embedding

        # Offset logits index and calculate cross entropy loss
        self.num_text_tokens = num_text_tokens
        self.linear_project = linear_project

        # Add <BOS> and <EOS> tokens to the beginning and end of text sequences
        if text_embedding.lower() in ("esm1b", "esm2"):
            self.text_seq_len = text_seq_len + 2
        else:
            self.text_seq_len = text_seq_len

        # Initialize embeddings for <SEP> token
        self.sep_emb = nn.Embedding(1, dim)

        # Initialize positional embeddings for text sequences and <SEP> token
        self.text_pos_emb = (
            nn.Embedding(self.text_seq_len + 1, dim) if not rotary_emb else always(0)
        )  # +1 for <SEP>

        ### ###

        self.num_images = num_images

        ### Initializing condition parameters ###

        # Initialize the number of condition tokens, condition sequence length, and condition embedding
        if exists(condition_vae):
            condition_size = condition_vae.image_size
            num_condition_tokens = condition_vae.num_tokens
            self.num_condition_tokens = num_condition_tokens
            condition_fmap_size = condition_vae.image_size // (
                2**condition_vae.num_layers
            )
            condition_seq_len = condition_fmap_size**2

            # Initialize ImageEmbedding for condition embedding
            self.condition_emb = ImageEmbedding(num_condition_tokens + 1, dim)

            # Initialize positional embeddings for condition embedding
            self.condition_pos_emb = (
                AxialPositionalEmbedding(
                    dim, axial_shape=(condition_fmap_size, condition_fmap_size)
                )
                if not rotary_emb
                else always(0)
            )

        else:
            condition_fmap_size = 0
            condition_seq_len = 0
            num_condition_tokens = 0

        ### ####

        ### Initializing image parameters ###

        # Initialize the image size, image token size, and sequence length
        self.image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = vae.image_size // (2**vae.num_layers)
        image_seq_len = image_fmap_size**2
        self.image_seq_len = image_seq_len
        self.num_image_tokens = num_image_tokens

        # Initialize ImageEmbedding and positional embeddings for image embedding
        self.image_emb = ImageEmbedding(num_image_tokens + 1, dim) # +1 for <IM_MASK>

        self.image_pos_emb = (
            AxialPositionalEmbedding(
                dim, axial_shape=(image_fmap_size, image_fmap_size)
            )
            if not rotary_emb
            else always(0)
        )

        # Set total sequence length and total tokens
        self.num_condition_tokens = num_condition_tokens
        self.condition_seq_len = condition_seq_len
        # Text Length + <SEP> + Condition Tokens + Image Tokens
        seq_len = self.text_seq_len + 1 + self.condition_seq_len + self.image_seq_len
        total_tokens = (
            num_text_tokens + 1 + num_condition_tokens + 1 + num_image_tokens + 1
        )
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        # Set the VAE and condition VAE for the model
        self.vae = vae.eval()
        self.condition_vae = condition_vae.eval()

        ### ###

        ### Setting discrete ids ###
        # Initialize text embedding based on the given text_embedding parameter
        if text_embedding == "esm1b" or text_embedding == "esm2":
            self.text_mask_token = 32
            self.pad_token = 1
            self.text_emb = ModelExtender(text_embedding, dim, fixed_embedding)
        else:
            raise ValueError("Only ESM models are supported.")

        # Set token indices for text, condition, and image sequences
        self.sep_token = num_text_tokens
        self.cond_mask_token = num_condition_tokens
        self.image_mask_token = num_image_tokens

        # Create indices for sequence and logits dimensions
        self.seq_range = torch.arange(seq_len)
        self.logits_range = torch.arange(total_tokens)

        # Reshape sequence and logits indices
        self.seq_range = rearrange(self.seq_range, "n -> () n ()")
        self.logits_range = rearrange(self.logits_range, "d -> () () d")

        # Create a mask to exclude invalid token positions from the model output
        # e.g. no image tokens where sequence tokens should be
        logits_mask = (
            # Mask text tokens beyond text_seq_len and invalid logits_range
            (
                (self.seq_range < self.text_seq_len)
                & (self.logits_range < num_text_tokens)
                & (self.logits_range != self.text_mask_token)
            )
            |
            # Mask [SEP] token after text
            (
                (self.seq_range == self.text_seq_len)
                & (self.logits_range == num_text_tokens)
            )
            |
            # Mask condition tokens beyond text_seq_len+1 ([SEP]) and invalid logits_range
            (
                (self.seq_range >= self.text_seq_len + 1)
                & (self.seq_range < self.text_seq_len + 1 + condition_seq_len)
                & (self.logits_range >= num_text_tokens + 1)
                & (self.logits_range < num_text_tokens + 1 + num_condition_tokens)
            )
            |
            # Mask image tokens beyond num_text_tokens+num_condition_tokens+1
            (
                (self.seq_range >= self.text_seq_len + 1 + condition_seq_len)
                & (self.logits_range >= num_text_tokens + 1 + num_condition_tokens + 1)
                & (
                    self.logits_range
                    < num_text_tokens + 1 + num_condition_tokens + 1 + num_image_tokens
                )
            )
        )

        # Invert the mask
        logits_mask = ~logits_mask

        # Register the buffer with the logits_mask
        self.register_buffer("logits_mask", logits_mask, persistent=False)

        ### ###

        # Initialize the Transformer model with given parameters
        self.transformer = Transformer(
            dim=dim,
            causal=causal,
            seq_len=seq_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            image_fmap_size=image_fmap_size + condition_fmap_size,
            num_images=num_images,
            stable=stable,
            rotary_emb=rotary_emb,
        )

        # Initialize the linear layers for converting transformer output to logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        # Set instance variables for weights and critic
        self.loss_img_weight = loss_img_weight
        self.loss_cond_weight = loss_cond_weight
        self.gamma = gamma_func(sampling_mode)

    def embed_and_transform(self, inputs, masks, return_encoding=False):
        text, condition, image = inputs
        device = text.device
        text_mask, _, image_mask = masks

        text_labels = text.clone()
        text = torch.where(
            text_mask, self.text_mask_token * torch.ones_like(text, device=device), text
        )

        tokens = self.text_emb(text)

        # Add SEP token

        sep_token_emb = self.sep_emb(
            torch.zeros((tokens.shape[0], 1), dtype=torch.long, device=device)
        )
        tokens = torch.cat((tokens, sep_token_emb), dim=1)
        tokens += self.text_pos_emb(torch.arange(text.shape[1] + 1, device=device))

        with torch.no_grad():
            if self.linear_project:
                b = condition.shape[0]
                condition, _, [_, _, condition_labels] = self.condition_vae.encode(
                    condition
                )
                condition_labels = rearrange(condition_labels, "(b n) -> b n", b=b)

            else:
                condition_labels = condition
                if condition.dtype == torch.float:
                    condition_labels = self.condition_vae.get_codebook_indices(
                        condition
                    )
                condition = condition_labels.clone()

        condition_emb = self.condition_emb(condition)
        condition_emb += self.condition_pos_emb(condition_emb)
        tokens = torch.cat((tokens, condition_emb), dim=1)

        with torch.no_grad():
            if self.linear_project:
                b = image.shape[0]
                image, _, [_, _, image_labels] = self.vae.encode(image)
                image_labels = rearrange(image_labels, "(b n) -> b n", b=b)

            else:
                image_labels = image
                if image.dtype == torch.float:
                    image_labels = self.vae.get_codebook_indices(image)
                image = torch.where(
                    image_mask,
                    self.image_mask_token
                    * torch.ones_like(image_labels, device=device),
                    image_labels,
                )

        image_emb = self.image_emb(image)

        image_emb += self.image_pos_emb(image_emb)
        tokens = torch.cat((tokens, image_emb), dim=1)

        if self.stable:
            alpha = 0.1
            tokens = tokens * alpha + tokens.detach() * (1 - alpha)

        out = self.transformer(tokens)

        if self.stable:
            out = self.norm_by_max(out)

        logits = self.to_logits(out)

        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(self.logits_mask, max_neg_value)

        if return_encoding:
            return logits, out, [text_labels, condition_labels, image_labels]
        else:
            return logits, None, [text_labels, condition_labels, image_labels]

    def forward(
        self,
        text,
        condition=None,
        image=None,
        return_loss=False,
        return_encoding=False,
    ):
        batch_size, device = text.shape[0], text.device

        # Check that image is supplied when training
        assert exists(image), "when training, image must be supplied"

        # Check that image dimensions match the expected dimensions
        assert tuple(image.shape[1:]) == (
            self.vae.channels,
            self.image_size,
            self.image_size,
        ), f"invalid image of dimensions {image.shape} passed in during training"

        # Generate masks for text, condition, and image

        # text_mask = generate_mask(self.gamma, batch_size, self.text_seq_len, device)

        text_mask = generate_mask(
            gamma_func("scaled-cosine"), batch_size, self.text_seq_len, device
        )

        image_mask = generate_mask(self.gamma, batch_size, self.image_seq_len, device)

        # Embed and transform inputs
        logits, _, labels = self.embed_and_transform(
            [text, condition, image],
            [text_mask, None, image_mask],
            return_encoding,
            device,
        )

        # If not returning loss, return the logits
        if not return_loss:
            return logits

        # Separate labels
        text, condition, image = labels

        # Add SEP token to end of text label
        sep_token = torch.tensor(self.sep_token, device=device).repeat(
            labels.shape[0], 1
        )
        labels = torch.cat([labels, sep_token], dim=1)

        # If condition exists and condition vae is defined, add the condition to the labels
        if exists(condition) and exists(self.condition_vae):
            offsetted_condition = condition + self.num_text_tokens + 1
            labels = torch.cat((labels, offsetted_condition), dim=1)

        # Add image to the labels
        offsetted_image = (
            image + self.num_text_tokens + 1 + self.num_condition_tokens + 1
        )
        labels = torch.cat((labels, offsetted_image), dim=1)

        # Rearrange logits for cross-entropy loss calculation
        # Logits size: (batch_size, vocab_size, total_seq_len)
        # Labels size: (batch_size, total_seq_len)
        logits = rearrange(logits, "b n c -> b c n")

        # Calculate cross-entropy loss for text and image
        loss_text = F.cross_entropy(
            logits[:, :, : self.text_seq_len],
            labels[:, : self.text_seq_len],
            reduction="none",
        )[text_mask].mean()

        loss_img = F.cross_entropy(
            logits[:, :, self.text_seq_len + 1 + self.condition_seq_len :],
            labels[:, self.text_seq_len + 1 + self.condition_seq_len :],
            reduction="none",
        )[image_mask].mean()

        # Calculate total loss
        loss = (loss_text + self.loss_img_weight * loss_img) / (
            self.loss_img_weight + 1
        )

        loss_dict = {
            "loss_text": loss_text,
            # "loss_cond": loss_cond,
            "loss_img": loss_img,
            "loss": torch.nan_to_num(loss, 0.0, 0.0, 0.0),
        }

        return loss, loss_dict, None

    def create_tensors(self, text, condition, image):
        """
        This function creates tensors for text, condition, and image when they are not provided as inputs to the sample function.
        """
        device = next(
            filter(lambda x: isinstance(x, torch.Tensor), [text, condition, image]),
            None,
        ).device

        if not isinstance(text, torch.Tensor):
            text = (
                torch.ones(1, self.text_seq_len, device=device, dtype=torch.long)
                * self.text_mask_token
            )

        if not isinstance(condition, torch.Tensor):
            condition = (
                torch.ones(1, self.condition_seq_len, device=device, dtype=torch.long)
                * self.cond_mask_token
            )
        else:
            with torch.no_grad():
                condition = self.condition_vae.get_codebook_indices(condition)

        if not isinstance(image, torch.Tensor):
            image = (
                torch.ones(1, self.image_seq_len, device=device, dtype=torch.long)
                * self.image_mask_token
            )
        else:
            with torch.no_grad():
                image = self.vae.get_codebook_indices(image)

        return text, condition, image

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        text=None,
        condition=None,
        image=None,
        temperature=1.0,
        filter_thres=0.9,
        progress=False,
        timesteps=1,
        force_aas=True,
    ):
        # ensure timesteps is a positive integer
        assert int(timesteps) > 0
        # set model and VAEs to evaluation mode
        self.eval()
        vae = self.vae.eval()
        if progress == True:
            progress = tqdm
        else:
            progress = lambda x: x


        # ensure that at least one of text, condition, or image is supplied
        assert (
            isinstance(text, torch.Tensor)
            or isinstance(condition, torch.Tensor)
            or isinstance(image, torch.Tensor)
        ), "some data must be supplied"

        # convert text, condition, and image to tensors if they aren't already
        text, condition, image = self.create_tensors(text, condition, image)

        # determine the maximum batch size of the input tensors
        batch_size = max(text.shape[0], condition.shape[0], image.shape[0])

        # match the batch sizes of text, condition, and image
        text, condition, image = match_batch_size(text, condition, image, batch_size)

        # determine the device of the tensors
        device = next(
            filter(lambda x: isinstance(x, torch.Tensor), [text, condition, image]),
            None,
        ).device

        assert text.shape[0] == condition.shape[0] == image.shape[0]

        # Create a tensor of zeros of size (batch_size, image_seq_len, num_image_tokens + 1) and set it to device

        # full_text_logits = torch.zeros(batch_size, self.text_seq_len, self.num_text_tokens+3).to(device)
        full_text_logits = torch.zeros(
            batch_size, self.text_seq_len, self.num_text_tokens
        ).to(device)

        # Use scatter_ to fill the tensor with 1 values at the indices given by the image tensor
        full_text_logits = full_text_logits.scatter_(
            dim=-1, index=text.unsqueeze(-1), value=1
        )
        # Use scatter_ to fill the tensor with 1 values at the indices given by the image tensor
        full_image_logits = torch.zeros(
            batch_size, self.image_seq_len, self.num_image_tokens + 1
        ).to(device)

        # Remove the last token from each image sequence by setting full_image_logits to its first num_image_tokens elements
        full_image_logits = full_image_logits.scatter_(
            dim=-1, index=image.unsqueeze(-1), value=1
        )

        # cut off mask token
        full_image_logits = full_image_logits[:, :, : self.num_image_tokens]

        count = 0

        for timestep in progress(torch.linspace(0, 1, timesteps)):
            # Create masks for the text, condition, and image tensors
            text_mask = text == self.text_mask_token
            cond_mask = condition == self.cond_mask_token
            image_mask = image == self.image_mask_token

            # Calculate logits and samples using the calculate_logits function
            logits, sample = calculate_logits(
                [text, condition, image],
                [text_mask, cond_mask, image_mask],
                self.embed_and_transform,
                filter_thres,
                temperature,
            )

            # Calculate the number of masked tokens in the text and image tensors
            num_masked_text_tokens = torch.sum(text_mask, dim=1)[0]
            num_masked_image_tokens = torch.sum(image_mask, dim=1)[0]

            # If there are masked text tokens, unmask them using unmask_tokens and fill the full text logits tensor with -inf for unmasked tokens
            if num_masked_text_tokens.any() > 0:
                text, full_text_logits = unmask_tokens(
                    text,
                    text_mask,
                    num_masked_text_tokens,
                    logits[:, : self.text_seq_len, : self.num_text_tokens],
                    sample[:, : self.text_seq_len, : self.num_text_tokens],
                    timestep,
                    timesteps,
                    self.gamma,
                    suppress_invalid_text_tokens,
                    self.pad_token,
                    self.text_mask_token,
                    force_aas=force_aas,
                )
                full_text_logits = full_text_logits.masked_fill(
                    ~text_mask.unsqueeze(-1), -torch.inf
                )

            # If there are masked image tokens, unmask them using unmask_tokens and fill the full image logits tensor with -inf for unmasked tokens
            if num_masked_image_tokens > 0:
                image, full_image_logits = unmask_tokens(
                    image,
                    image_mask,
                    num_masked_image_tokens,
                    logits[:, -self.image_seq_len :, -(self.num_image_tokens + 1) : -1],
                    sample[:, -self.image_seq_len :, -(self.num_image_tokens + 1) : -1],
                    timestep,
                    timesteps,
                    self.gamma,
                )
                full_text_logits = full_text_logits.masked_fill(
                    ~text_mask.unsqueeze(-1), -torch.inf
                )
                
        # Generate heatmap
        with torch.no_grad():
            # Normalize full image logits tensor
            full_image_logits /= torch.max(
                torch.abs(full_image_logits), dim=-1, keepdim=True
            ).values

            # Apply quantize embedding to full image logits tensor
            full_image_logits = torch.matmul(
                full_image_logits, self.vae.model.quantize.embedding.weight
            )

            # Rearrange full image logits tensor
            h = int(self.image_seq_len**0.5)
            full_image_logits = rearrange(
                full_image_logits, "b (h w) c -> b c h w", h=h
            )

            # Decode full image logits tensor
            full_image_logits = self.vae.model.decode(full_image_logits)

            # Add clipping to full image logits tensor
            max_val = torch.max(full_image_logits.view(batch_size, -1), dim=-1)[0]
            min_val = torch.min(full_image_logits.view(batch_size, -1), dim=-1)[0]
            full_image_logits += torch.clip(1 - max_val, 0, float("inf")).view(
                batch_size, 1, 1, 1
            )
            full_image_logits += torch.clip(0 - min_val, float("-inf"), 0).view(
                batch_size, 1, 1, 1
            )

            # Clip full image logits tensor values to the range [0, 1]
            full_image_logits = torch.clip(full_image_logits, 0, 1)

        # Return text tensor, detokenized text tensor, full text logits tensor,
        # binary image tensor, and full image logits tensor
        return (
            text,
            detokenize_text(self.text_embedding, text),
            full_text_logits,
            1.0 * (vae.decode(image) > 0.5),
            full_image_logits,
        )

    @torch.no_grad()
    @eval_decorator
    def sample_text(
        self,
        text=False,
        condition=False,
        image=False,
        temperature=1.0,
        filter_thres=0.9,
        progress=False,
        n_unmask=1,
        place_amino=True,
        force_aas=False,
    ):
        # set model and VAEs to evaluation mode
        self.eval()

        # ensure that at least one of text, condition, or image is supplied
        assert (
            isinstance(text, torch.Tensor)
            or isinstance(condition, torch.Tensor)
            or isinstance(image, torch.Tensor)
        ), "some data must be supplied"

        # convert text, condition, and image to tensors if they aren't already
        text, condition, image = self.create_tensors(text, condition, image)

        # determine the maximum batch size of the input tensors
        batch_size = max(text.shape[0], condition.shape[0], image.shape[0])

        # match the batch sizes of text, condition, and image
        text, condition, image = match_batch_size(text, condition, image, batch_size)

        # determine the device of the tensors
        device = next(
            filter(lambda x: isinstance(x, torch.Tensor), [text, condition, image]),
            None,
        ).device

        assert text.shape[0] == condition.shape[0] == image.shape[0]

        # Create a tensor of zeros of size (batch_size, image_seq_len, num_image_tokens + 1) and set it to device

        # full_text_logits = torch.zeros(batch_size, self.text_seq_len, self.num_text_tokens+3).to(device)
        full_text_logits = torch.zeros(
            batch_size, self.text_seq_len, self.num_text_tokens
        ).to(device)

        # Use scatter_ to fill the tensor with 1 values at the indices given by the image tensor
        full_text_logits = full_text_logits.scatter_(
            dim=-1, index=text.unsqueeze(-1), value=1
        )

        text_mask = text == self.text_mask_token
        cond_mask = condition == self.cond_mask_token
        image_mask = image == self.image_mask_token

        mask_indices = text_mask.nonzero()
        non_mask_indices = (~text_mask).nonzero()

        # figure out the center of the amino acids to determine generation direction
        central_protein_index = torch.tensor(
            [
                torch.median(
                    non_mask_indices[torch.where(non_mask_indices[:, 0] == idx)][:, -1]
                )
                for idx in range(batch_size)
            ]
        )

        count = 1

        run_mask = text_mask
        if progress:
            pbar = progress(total=torch.sum(run_mask).item())
        while torch.sum(run_mask) > 0:
            logits, sample = calculate_logits(
                [text, condition, image],
                [text_mask, cond_mask, image_mask],
                self.embed_and_transform,
                filter_thres,
                temperature,
            )

            # sub_sample: [batch_size ,text_seq_len ,num_text_tokens]
            sub_sample = sample[:, : self.text_seq_len, : self.num_text_tokens]
            sub_sample = sub_sample.masked_fill(~text_mask.unsqueeze(-1), -torch.inf)
            sub_sample = suppress_invalid_text_tokens(
                text, sub_sample, 0, 2, self.pad_token, self.text_mask_token, force_aas
            )
            # calculate % to  unmasked
            # get most likely token and probability for each position

            for idx in range(batch_size):
                selected_mask_indices = mask_indices[
                    torch.where(mask_indices[:, 0] == idx)
                ][:, -1]

                # Generate to the left
                if selected_mask_indices[-count] < central_protein_index[idx]:
                    unmask_index = selected_mask_indices[-count]
                    left_sample = max(0, (unmask_index + 1) - n_unmask)
                    right_sample = min(unmask_index + 1, self.text_seq_len - 1)
                    central_protein_index[idx] = max(
                        0, central_protein_index[idx] - 0.5 * n_unmask
                    )

                # Generate to the right
                elif selected_mask_indices[count - 1] > central_protein_index[idx]:
                    unmask_index = selected_mask_indices[count - 1]
                    left_sample = max(0, unmask_index)
                    right_sample = min(unmask_index + n_unmask, self.text_seq_len - 1)
                    central_protein_index[idx] = min(
                        central_protein_index[idx] + 0.5 * n_unmask,
                        self.text_seq_len - 1,
                    )

                # save logits for relevant position
                full_text_logits[
                    idx, left_sample:right_sample, : self.text_seq_len - 1
                ] = logits[idx, left_sample:right_sample, : self.num_text_tokens]

                run_mask[idx, left_sample:right_sample] = False

                # you may want to resample the amion acids or calculate marginal probs
                # if so, set place_amino to false
                if place_amino:
                    text[idx, left_sample:right_sample] = torch.where(
                        text[idx, left_sample:right_sample] == self.text_mask_token,
                        sub_sample[
                            idx, left_sample:right_sample, : self.num_text_tokens
                        ].argmax(dim=-1),
                        text[idx, left_sample:right_sample],
                    )

                    text_mask = run_mask

            count += n_unmask
            
            if progress:
                pbar.update(n_unmask)
        if progress:
            pbar.close()

        return (
            text,
            detokenize_text(self.text_embedding, text),
            full_text_logits,
        )
