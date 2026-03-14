import math
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from Plug_and_Play_Attacks.losses.poincare import poincare_loss

class Optimization():
    def __init__(self, target_model, synthesis, discriminator, transformations, num_ws, config):
        self.synthesis = synthesis
        self.target = target_model
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.num_ws = num_ws
        self.clip = config.attack['clip']

        attack_loss_function = self.config.attack.get('attack_loss_function', 'poincare')
        if attack_loss_function == 'poincare':
            self.attack_loss = poincare_loss
        elif attack_loss_function == 'cross_entropy':
            self.attack_loss = nn.CrossEntropyLoss()
        elif attack_loss_function == 'logit_loss':
            self.attack_loss = None #! TODO

        #! for avg conf tracking
        self.conf_dict = {} # conf_dict[class] = [list of average target confidences for that class during optimization]

    def optimize(self, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        gif_folder = 'gifs'
        gif_save_freq = 1
        
        if gif_folder:
            os.makedirs(gif_folder, exist_ok=True)

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            if gif_folder and (i % gif_save_freq == 0):
                with torch.no_grad():
                    self._save_gif_frame(imgs, targets_batch, gif_folder, i)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(
                    imgs)
            else:
                discriminator_loss = torch.tensor(0.0)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute outputs
            outputs = self.target(imgs)

            # Compute target loss
            target_loss = self.attack_loss(
                    outputs, targets_batch).mean() # outputs: (bsz, num_classes), targets_batch: (bsz,)


            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # Log results
            if self.config.log_progress:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(
                        confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()

                if torch.cuda.current_device() == 0:
                    print(
                        f'iteration {i}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f}'
                    )

                if i == num_epochs - 1: #! calculate AvgTargetConf
                    for curr_target in set(targets_batch.cpu().tolist()):
                        if curr_target not in self.conf_dict:
                            self.conf_dict[curr_target] = []
                        target_indices = (targets_batch == curr_target).nonzero(as_tuple=True)[0]
                        curr_target_mean_conf = confidences[target_indices].mean().item()
                        self.conf_dict[curr_target].append(curr_target_mean_conf)

        return w_batch.detach()

    def synthesize(self, w, num_ws):
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            imgs = self.synthesis(w_expanded,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis(w, noise_mode='const', force_fp32=True)
        return imgs

    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def _save_gif_frame(self, imgs, targets_batch, folder, iteration, resize=256):
        unique_targets = sorted(set(targets_batch.cpu().tolist()))
        cols = []
        for t in unique_targets:
            idx = (targets_batch == t).nonzero(as_tuple=True)[0][0]
            img = imgs[idx].detach().cpu()
            img = (img * 0.5 + 128 / 255).clamp(0, 1)
            img = TF.resize(img, resize, antialias=True)
            cols.append(img)
        frame = torch.cat(cols, dim=2)
        path = os.path.join(folder, f'frame_{iteration:05d}.png')
        torchvision.utils.save_image(frame, path)

    def compute_discriminator_loss(self, imgs):
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(
            -discriminator_logits).mean()
        return discriminator_loss