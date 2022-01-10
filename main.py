import os
import argparse
import yaml
from tqdm import tqdm
import torch
from torch import nn
from torchvision.utils import make_grid

from utils.utils import get_device, get_logger, get_tb_writer, get_optimizer, get_generator_net, \
    get_discriminator_net, save_model
from utils.dataset import create_dataloader


class Trainer():

    def __init__(self, config):
        self.logger = get_logger()
        self.data = config["Data"]["data_path"]
        self.input_dim = config["Training"]["input_dim"]
        self.channels = config["Training"]["channels"]
        self.batch_size = config["Training"]["batch_size"]
        self.image_size = config["Training"]["image_size"]
        self.epochs = config["Training"]["epochs"]
        self.device = get_device(config["Training"]["device"])
        self.tb_writer = get_tb_writer(config["Logging"]["train_logs"])
        self.checkpints_dir = os.path.join(config["Logging"]["train_logs"], "checkpoints")
        self.train_dataloader = create_dataloader(self.data, self.image_size, self.batch_size, self.channels)
        self.generator_net = get_generator_net(self.input_dim, self.channels, self.device,
                                               config["Generator"]["pretrained_weights"])
        self.discriminator_net = get_discriminator_net(self.channels, self.device,
                                                       config["Generator"]["pretrained_weights"])
        self.g_optimizer = get_optimizer(self.generator_net, config["Generator"]["optimizer"],
                                         config["Generator"]["learning_rate"])
        self.d_optimizer = get_optimizer(self.generator_net, config["Discriminator"]["optimizer"],
                                         config["Discriminator"]["learning_rate"])
        self.criterion = nn.BCELoss()

    def train(self):
        global_step = 0  # Log metrics to tensorboard for every batch
        log_step = int(len(self.train_dataloader) / 3)  # Display loss on the std output every log_step batches
        total_batches = len(self.train_dataloader)
        for epoch in range(self.epochs):
            epoch += 1
            pbar = tqdm(self.train_dataloader)
            for real_images in pbar:
                # Define labels for real and generated images
                real_images_labels = torch.ones((real_images.shape[0],), device=self.device)
                generated_images_labels = torch.zeros((real_images.shape[0],), device=self.device)
                batch = global_step - (epoch-1) * total_batches + 1
                pbar.set_description(f"epoch: {epoch}/{self.epochs}, batch: {batch}/{total_batches}")
                real_images = real_images.cuda(self.device)
                self.d_optimizer.zero_grad()

                ###  Discriminator learning  ###
                # Get discriminator loss for real images from dataset
                discriminator_pred_real = self.discriminator_net(real_images)

                discriminator_loss_real = self.criterion(discriminator_pred_real, real_images_labels)
                # Get discriminator loss for generated images from generator network
                input_noise = torch.randn((real_images.shape[0], self.input_dim), device=self.device)
                generated_images = self.generator_net(input_noise)
                generated_images = generated_images.cuda(self.device)
                discriminator_pred_generated = self.discriminator_net(generated_images)
                discriminator_loss_generated = self.criterion(discriminator_pred_generated, generated_images_labels)
                discriminator_loss = discriminator_loss_real + discriminator_loss_generated
                discriminator_loss.backward()
                self.d_optimizer.step()

                ###  Generator learning  ###
                self.g_optimizer.zero_grad()
                input_noise = torch.randn((real_images.shape[0], self.input_dim), device=self.device)
                generated_images = self.generator_net(input_noise)
                discriminator_pred_generated = self.discriminator_net(generated_images)
                # Calculate loss for generated images and labels for real images
                generator_loss = self.criterion(discriminator_pred_generated, real_images_labels)
                generator_loss.backward()
                self.g_optimizer.step()

                # Log loss and learning rate
                self.tb_writer.add_scalar("Generator Loss", generator_loss.item(), global_step)
                self.tb_writer.add_scalar("Discriminator Loss", discriminator_loss.item(), global_step)
                self.tb_writer.add_scalar("Generator Learning Rate", self.g_optimizer.param_groups[0]["lr"],
                                          global_step)
                self.tb_writer.add_scalar("Discriminator Learning Rate", self.d_optimizer.param_groups[0]["lr"],
                                          global_step)
                global_step += 1

                if global_step % log_step == 0:
                    dis_pred_real = (discriminator_pred_real > 0.5).type(torch.float32)
                    acc_real = float(torch.mean((dis_pred_real == real_images_labels).type(torch.float32)))
                    dis_pred_fake = (discriminator_pred_generated > 0.5).type(torch.float32)
                    acc_fake = float(torch.mean((dis_pred_fake == generated_images_labels).type(torch.float32)))
                    self.logger.info(f"Epoch: {epoch}, Batch: {batch}, global step: {global_step}, Generator Loss: {generator_loss.item()} "
                                     f"Discriminator Loss: {discriminator_loss.item()}, Discriminator Accuracy Real: {acc_real}, "
                                     f"Discriminator Accuracy Fake: {acc_fake}")
                    generated_image = self.generator_net(input_noise)
                    generated_image_grid = make_grid(generated_image)
                    self.tb_writer.add_image("generated_images", generated_image_grid, global_step)
            # Save models
            save_model(self.generator_net, "generator", epoch, self.checkpints_dir)
            save_model(self.discriminator_net, "discriminator", epoch, self.checkpints_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='',
                        help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    trainer = Trainer(config)
    trainer.train()

    # genNet = Generator(100, 3)
    # genOut = genNet(input_noise)
    #
    # disNet = Discriminator(3)
    # disOut = disNet(genOut)
    # print("disOut: ", disOut.shape)
    # print(disOut)
