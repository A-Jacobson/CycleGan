import itertools
import os

import torch
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from config import (PATH_A, PATH_B, NUM_EPOCHS,
                    SCALE, CYCLE_WEIGHT, WEIGHTS_DIR, GAN_CRIT,
                    RESUME_EPOCH, VISUALIZATION_FREQ, IDENTITY_WEIGHT,
                    CHECKPOINT_FREQ, EXP_NAME, NUM_RESBLOCKS)
from criterion import PatchGanLoss
from datasets import ImageDataset
from models import ResnetGenerator, PatchDiscriminator
from utils import to_var, to_tensor, to_image, AverageMeter, ImageHistory

cudnn.benchmark = True

transform = transforms.Compose([transforms.Scale(SCALE),
                                transforms.CenterCrop(SCALE),
                                transforms.ToTensor()])

dset_A = ImageDataset(PATH_A, transforms=transform)
dset_B = ImageDataset(PATH_B, transforms=transform)

loader_A = DataLoader(dset_A, shuffle=True, batch_size=1, pin_memory=True)
loader_B = DataLoader(dset_B, shuffle=True, batch_size=1, pin_memory=True)

# define generators
generator_A2B = ResnetGenerator(NUM_RESBLOCKS)
generator_B2A = ResnetGenerator(NUM_RESBLOCKS)

# define discriminators
discriminator_A = PatchDiscriminator()  # looks at output of b_to_a
discriminator_B = PatchDiscriminator()  # looks at output of a_to_b

g_params = itertools.chain(generator_A2B.parameters(), generator_B2A.parameters())
d_params = itertools.chain(discriminator_A.parameters(), discriminator_B.parameters())

g_optimizer = Adam(g_params, lr=2e-3)  # constant for 100 epochs, linear decay for 100 epochs
d_optimizer = Adam(d_params, lr=1e-3)

d_criterion = PatchGanLoss(criterion=GAN_CRIT)
cycle_criterion = nn.L1Loss()  # cycle loss is just L1 loss between original image and reconstructed image

if IDENTITY_WEIGHT:
    identity_criterion = nn.L1Loss()

if RESUME_EPOCH:
    generator_A2B.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'generator_A2B_{}.pkl'.format(RESUME_EPOCH))))
    generator_B2A.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'generator_B2A_{}.pkl'.format(RESUME_EPOCH))))
    discriminator_A.load_state_dict(
        torch.load(os.path.join(WEIGHTS_DIR, 'discriminator_A_{}.pkl'.format(RESUME_EPOCH))))
    discriminator_B.load_state_dict(
        torch.load(os.path.join(WEIGHTS_DIR, 'discriminator_B_{}.pkl'.format(RESUME_EPOCH))))

    g_optimizer.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'g_optimizer_{}.pkl'.format(RESUME_EPOCH))))
    d_optimizer.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'd_optimizer_{}.pkl'.format(RESUME_EPOCH))))

num_batches = min(len(loader_A), len(loader_B))

fake_A_history = ImageHistory(max_length=50)  # image buffers to improve discriminator stability
fake_B_history = ImageHistory(max_length=50)

for epoch in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS):

    if torch.cuda.is_available():
        generator_A2B.cuda()
        generator_B2A.cuda()
        discriminator_A.cuda()
        discriminator_B.cuda()

    current_epoch = RESUME_EPOCH + (epoch + 1)
    d_loss_meter = AverageMeter()
    g_loss_meter = AverageMeter()
    pbar = tqdm(range(num_batches), total=num_batches)

    for batch in pbar:
        img_A = next(iter(loader_A))
        img_B = next(iter(loader_B))
        img_A = to_var(img_A)
        img_B = to_var(img_B)

        # TRAIN DISCRIMINATORS
        # loss for A discriminator
        fake_A = generator_B2A(img_B)
        fake_A = fake_A.detach()  # detach generator output because we don't need generator gradients yet
        fake_A_history.update(fake_A)  # update buffer of generated images
        fake_A_old = fake_A_history.sample()
        disc_A_batch = torch.cat([img_A, fake_A, fake_A_old])  # train with real, current fake, old fake
        disc_A_targets = torch.Tensor([1, 0, 0])  # manually create targets for discriminator
        disc_A_score = discriminator_A(disc_A_batch)
        disc_A_loss = d_criterion(disc_A_score, disc_A_targets)

        # loss for zebra discriminator
        fake_B = generator_A2B(img_A)
        fake_B = fake_B.detach()
        fake_B_history.update(fake_B)
        fake_B_old = fake_B_history.sample()
        disc_B_batch = torch.cat([img_B, fake_B, fake_B_old])
        disc_B_targets = torch.Tensor([1, 0, 0])
        disc_B_score = discriminator_B(disc_B_batch)
        disc_B_loss = d_criterion(disc_B_score, disc_B_targets)

        # add up losses, optimize discriminators
        d_loss = disc_A_loss + disc_B_loss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # TRAIN GENERATORS
        d_loss_target = torch.Tensor([1])  # we now want the score for the generated animals to be one

        fake_B = generator_A2B(img_A)
        disc_B_score = discriminator_A(fake_B)
        reconstructed_A = generator_B2A(fake_B)
        disc_B_loss = d_criterion(disc_B_score, d_loss_target)
        cycle_loss_A = cycle_criterion(reconstructed_A, img_A)

        fake_A = generator_B2A(img_B)
        disc_A_score = discriminator_B(fake_A)
        reconstructed_B = generator_A2B(fake_A)
        disc_A_loss = d_criterion(disc_A_score, d_loss_target)
        cycle_loss_B = cycle_criterion(reconstructed_B, img_B)

        # enforce color consistency between original and generated images for non-color transforms
        identity_loss = 0
        if IDENTITY_WEIGHT:
            identity_loss_A = identity_criterion(fake_A, img_B) * IDENTITY_WEIGHT
            identity_loss_B = identity_criterion(fake_B, img_A) * IDENTITY_WEIGHT
            identity_loss = identity_loss_A + identity_loss_B

        g_loss = disc_B_loss + disc_A_loss + (CYCLE_WEIGHT * (cycle_loss_B + cycle_loss_A + identity_loss))

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        g_loss_meter.update(g_loss.data[0])
        d_loss_meter.update(d_loss.data[0])

        pbar.set_description("[ d_loss: {:.4f} | g_loss: {:.4f} ]".format(d_loss_meter.avg, g_loss_meter.avg))

        # CHECKPOINTS

        if (epoch + 1) % VISUALIZATION_FREQ == 0 and (batch + 1) == num_batches:
            if not os.path.exists(os.path.join('outputs', EXP_NAME, 'ABA')):
                os.makedirs(os.path.join('outputs', EXP_NAME, 'ABA'))

            if not os.path.exists(os.path.join('outputs', EXP_NAME, 'BAB')):
                os.makedirs(os.path.join('outputs', EXP_NAME, 'BAB'))

            img_A = to_tensor(img_A)
            fake_B = to_tensor(fake_B)
            reconstructed_A = to_tensor(reconstructed_A)
            cycle_ABA = make_grid(torch.cat([img_A, fake_B, reconstructed_A]), nrow=3)
            cycle_ABA = to_image(cycle_ABA)
            cycle_ABA.save(os.path.join('outputs', EXP_NAME, 'ABA', '{}.jpg'.format(current_epoch)))

            img_B = to_tensor(img_B)
            fake_A = to_tensor(fake_A)
            reconstructed_B = to_tensor(reconstructed_B)
            cycle_BAB = make_grid(torch.cat([img_B, fake_A, reconstructed_B]), nrow=3)
            cycle_BAB = to_image(cycle_BAB)
            cycle_BAB.save(os.path.join('outputs', EXP_NAME, 'BAB', '{}.jpg'.format(current_epoch)))

    if (epoch + 1) % CHECKPOINT_FREQ == 0 or (epoch + 1) == NUM_EPOCHS:
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)

        torch.save(generator_A2B.cpu().state_dict(),
                   os.path.join(WEIGHTS_DIR, 'generator_A2B_{}.pkl'.format(current_epoch)))
        torch.save(generator_B2A.cpu().state_dict(),
                   os.path.join(WEIGHTS_DIR, 'generator_B2A_{}.pkl'.format(current_epoch)))
        torch.save(discriminator_A.cpu().state_dict(),
                   os.path.join(WEIGHTS_DIR, 'discriminator_A_{}.pkl'.format(current_epoch)))
        torch.save(discriminator_B.cpu().state_dict(),
                   os.path.join(WEIGHTS_DIR, 'discriminator_B_{}.pkl'.format(current_epoch)))

        torch.save(g_optimizer.state_dict(),
                   os.path.join(WEIGHTS_DIR, 'g_optimizer_{}.pkl'.format(current_epoch)))
        torch.save(d_optimizer.state_dict(),
                   os.path.join(WEIGHTS_DIR, 'd_optimizer_{}.pkl'.format(current_epoch)))
