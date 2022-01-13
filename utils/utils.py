import os
import json
import logging
import torch
from tensorboardX import SummaryWriter

from models.dcgan import Generator, Discriminator


def read_data(data_path):
    """
    :param data_path: Path to json file with list of images
    :return:
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def get_device(device):
    """
    Args:
        device: str, GPU device id
    Return: torch device
    """

    if device == "cpu":
        return torch.device("cpu")
    else:
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"
        c = 1024 ** 2
        x = torch.cuda.get_device_properties(0)
        print("Using GPU")
        print(f"device{device} _CudaDeviceProperties(name='{x.name}'"
              f", total_memory={x.total_memory / c}MB)")
        return torch.device("cuda:0")


def get_logger():
    logger = logging.getLogger("ClassNets")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s]-[%(filename)s]: %(message)s ")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def get_tb_writer(tb_logdir):
    """
    Args:
        tb_logdir: str, Path to directory fot tensorboard events
    Return:
        writer: TensorBoard writer
    """
    tb_logdir = os.path.join(tb_logdir, "tb_logs")
    if not os.path.isdir(tb_logdir):
        os.makedirs(tb_logdir)
    writer = SummaryWriter(log_dir=tb_logdir)
    return writer


def get_optimizer(model, opt, lr=0.001):
    """
    Args
        model: Model, generator or discriminator
        opt: string, optimizer from config file
        model: nn.Module, generated model
        lr: float, specified learning rate
    Returns:
        optimizer
    """

    if opt.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Not supported optimizer name: {opt}."
                                  f"For supported optimizers see documentation")
    return optimizer


def read_weights(model, pretrained_weights):
    """
    Args:
        model: Model
        pretrained_weights: Pretrained weights to read
    Returns:
        Model with pretrained weights
    """
    print(f"Loading weights from {pretrained_weights}...")
    model.load_state_dict(torch.load(pretrained_weights))
    model.eval()
    return model


def get_generator_net(input_dim, channels, device, pretrained_weights=''):
    """
    Args:
        input_dim: Input dimension to generator net
        channels: Number of channels of the image (output of the generator net)
        device: Device id
        pretrained_weights: Path to pretrained weights if exists
    Returns:
        Generator net
    """
    model = Generator(input_dim, channels)
    if pretrained_weights:
        model = read_weights(model, pretrained_weights)
    return model.train().cuda(device)


def get_discriminator_net(channels, device, pretrained_weights=''):
    """
    Args:
        channels: Number of channels of the input image
        device: Device id
        pretrained_weights: Path to pretrained weights if exists
    Returns:
        Discriminator net
    """
    model = Discriminator(channels)
    if pretrained_weights:
        model = read_weights(model, pretrained_weights)
    return model.train().cuda(device)


def save_model(model, model_name, epoch, ckpt_dir):
    """ Save model
    Args:
        model: Model for saving
        model_name: generator or discriminator
        epoch: Number of epoch
        ckpt_dir: Store directory
    """
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_epoch_" + str(epoch) + ".pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved: {ckpt_path}")


def get_random_input_vector(batch_size, input_dim, device):
    """
    Args:
        batch_size: Batch size
        input_dim: Input dimension of the generator net
        device: Device id
    Returns:
        Gaussian noise - random vector
    """
    return torch.randn((batch_size, input_dim), device=device)