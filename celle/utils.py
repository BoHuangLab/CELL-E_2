import torch
from math import pi

# Define helper functions
def exists(val):
    """Check if a variable exists"""
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    """If a value exists, return it; otherwise, return a default value"""
    return val if exists(val) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def cast_tuple(val, depth=1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth


def is_empty(t):
    """Check if a tensor is empty"""
    # Return True if the number of elements in the tensor is zero, else False
    return t.nelement() == 0


def masked_mean(t, mask, dim=1):
    """
    Compute the mean of a tensor, masked by a given mask

    Args:
        t (torch.Tensor): input tensor of shape (batch_size, seq_len, hidden_dim)
        mask (torch.Tensor): mask tensor of shape (batch_size, seq_len)
        dim (int): dimension along which to compute the mean (default=1)

    Returns:
        torch.Tensor: masked mean tensor of shape (batch_size, hidden_dim)
    """
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def set_requires_grad(model, value):
    """
    Set whether or not the model's parameters require gradients

    Args:
        model (torch.nn.Module): the PyTorch model to modify
        value (bool): whether or not to require gradients
    """
    for param in model.parameters():
        param.requires_grad = value


def eval_decorator(fn):
    """
    Decorator function to evaluate a given function

    Args:
        fn (callable): function to evaluate

    Returns:
        callable: the decorated function
    """

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def log(t, eps=1e-20):
    """
    Compute the natural logarithm of a tensor

    Args:
        t (torch.Tensor): input tensor
        eps (float): small value to add to prevent taking the log of 0 (default=1e-20)

    Returns:
        torch.Tensor: the natural logarithm of the input tensor
    """
    return torch.log(t + eps)


def gumbel_noise(t):
    """
    Generate Gumbel noise

    Args:
        t (torch.Tensor): input tensor

    Returns:
        torch.Tensor: a tensor of Gumbel noise with the same shape as the input tensor
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=0.9, dim=-1):
    """
    Sample from a Gumbel-softmax distribution

    Args:
        t (torch.Tensor): input tensor of shape (batch_size, num_classes)
        temperature (float): temperature for the Gumbel-softmax distribution (default=0.9)
        dim (int): dimension along which to sample (default=-1)

    Returns:
        torch.Tensor: a tensor of samples from the Gumbel-softmax distribution with the same shape as the input tensor
    """
    return (t / max(temperature, 1e-10)) + gumbel_noise(t)


def top_k(logits, thres=0.5):
    """
    Return a tensor where all but the top k values are set to negative infinity

    Args:
        logits (torch.Tensor): input tensor of shape (batch_size, num_classes)
        thres (float): threshold for the top k values (default=0.5)

    Returns:
        torch.Tensor: a tensor with the same shape as the input tensor, where all but the top k values are set to negative infinity
    """
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def gamma_func(mode="cosine", scale=0.15):
    """Return a function that takes a single input r and returns a value based on the selected mode"""

    # Define a different function based on the selected mode
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: torch.cos(r * pi / 2)
    elif mode == "square":
        return lambda r: 1 - r**2
    elif mode == "cubic":
        return lambda r: 1 - r**3
    elif mode == "scaled-cosine":
        return lambda r: scale * (torch.cos(r * pi / 2))
    else:
        # Raise an error if the selected mode is not implemented
        raise NotImplementedError


class always:
    """Helper class to always return a given value"""

    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return self.val


class DivideMax(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True).detach()
        return x / maxes

def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    if image_tensor.shape[0] > 1:
        image_tensor = torch.mean(image_tensor, dim=0, keepdim=True)
    return image_tensor.unsqueeze(0)
