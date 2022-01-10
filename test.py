import argparse
import torch
import numpy as np
import cv2 as cv

from models.dcgan import Generator
from utils.utils import get_device


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='',
                        help="Path to model")
    parser.add_argument("--in-channels", type=int, default=3,
                        help="Number of input channels")
    parser.add_argument("--input-dim", type=int, default=100,
                        help="Input dimension to generator network")
    parser.add_argument("--out-img", type=str, default='',
                        help="Path to output image")
    args = parser.parse_args()

    generator = Generator(args.input_dim, args.in_channels)
    generator.load_state_dict(torch.load(args.model))
    generator.eval()
    device = get_device('0')
    input_noise = torch.randn((1, args.input_dim), device=device)
    generated_image = generator(input_noise)
    generated_image = np.moveaxis(generated_image.detach().to('cpu').numpy()[0], 0, 2)
    # generated_image = np.repeat(generated_image, 3, axis=2)

    generated_image -= np.min(generated_image)
    generated_image /= np.max(generated_image)
    generated_image = generated_image * 128.
    print(generated_image.shape)
    cv.imwrite("gen_img.jpg", generated_image)
