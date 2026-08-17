import math
import numpy as np
import torch
import torch.nn as nn

from Plug_and_Play_Attacks.losses.poincare import poincare_loss
from Plug_and_Play_Attacks.losses.label_smoothing import CrossEntropyLoss as LabelSmoothingCrossEntropyLoss

#! BEGIN alignment score addition (see model_inversion/metrics/mi_gradient_alignment.py) - safe to delete this import + all blocks tagged "alignment score addition" to revert
from model_inversion.metrics.mi_gradient_alignment import compute_gradient_alignment_score, make_det_transformations
#! END alignment score addition

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
        elif attack_loss_function == 'negative_label_smoothing':
            # Negative label smoothing sharpens the target class further than
            # standard cross entropy, to (hopefully) counteract label-smoothing-based defenses.
            label_smoothing_factor = self.config.attack.get('label_smoothing_factor', -0.1)
            self.attack_loss = LabelSmoothingCrossEntropyLoss(label_smoothing=label_smoothing_factor)
        elif attack_loss_function == 'logit_loss':
            self.attack_loss = None #! TODO

        #! for avg target model confidence tracking
        self.conf_dict = {} # conf_dict[class] = [list of average target confidences for that class during optimization]

        #! for per-iteration target model confidence tracking (mirrors stdout logging)
        self.iter_conf_log = [] # list of [batch_idx, iteration, target_classes, mean_conf, grad_cosine_sim]
        self.batch_idx = -1

        #! BEGIN alignment score addition - safe to delete to revert
        alignment_score_config = self.config.attack.get('alignment_score', {})
        self.compute_alignment_score = alignment_score_config.get('compute', False)
        self.alignment_score_every = alignment_score_config.get('every', 10)
        self.alignment_score_chunk_size = alignment_score_config.get('chunk_size', 40)
        self.alignment_score_max_samples = alignment_score_config.get('max_samples', None)
        self.alignment_log = [] # list of [batch_idx, iteration, target_classes, mean_alignment_score]

        # Deterministic transforms: stochastic augmentations (RandomResizedCrop, ColorJitter, etc.)
        # stripped out so that both grad_x and the Jacobian are computed consistently.
        self.det_transformations = make_det_transformations(transformations)
        if transformations is not None:
            _all = transformations.transforms if hasattr(transformations, 'transforms') else [transformations]
            _kept_ids = set(id(t) for t in (self.det_transformations.transforms if hasattr(self.det_transformations, 'transforms') else ([self.det_transformations] if self.det_transformations else [])))
            _removed = [type(t).__name__ for t in _all if id(t) not in _kept_ids]
            if _removed:
                print(f"[alignment score] det_transformations: removed stochastic {_removed}")

        # raw (w, grad_x) pairs needed to compute the alignment score post-hoc, see
        # model_inversion/metrics/mi_gradient_alignment.compute_alignment_scores_from_raw_log.
        # Collected unconditionally (independent of compute_alignment_score) at the same cadence.
        # grad_x is computed via a separate deterministic forward pass (det_transformations).
        self.alignment_raw_log = [] # list of dicts: {batch_idx, iteration, target_classes, w, grad_x}
        #! END alignment score addition

    def optimize(self, w_batch, targets_batch, num_epochs):
        self.batch_idx += 1
        target_classes = sorted(set(targets_batch.cpu().tolist())) # logged alongside batch_idx/iteration below
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        #! BEGIN successive-gradient cosine similarity addition - safe to delete this block + the two tagged blocks below to revert
        # gradient of the loss w.r.t. the (stochastically transformed) input image from the previous
        # iteration, used to track how much the attack gradient direction shifts iteration to iteration
        prev_grad_x = None
        #! END successive-gradient cosine similarity addition

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

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

            #! BEGIN successive-gradient cosine similarity addition - safe to delete to revert
            # retain grad on the actual (stochastically transformed) model input so we can read
            # d(loss)/d(imgs) after backward(), without touching the alignment score's separate
            # deterministic gradient computation below
            if self.config.log_progress:
                imgs.retain_grad()
            #! END successive-gradient cosine similarity addition

            # Compute outputs
            outputs = self.target(imgs)

            # Compute target loss
            target_loss = self.attack_loss(
                    outputs, targets_batch).mean() # outputs: (bsz, num_classes), targets_batch: (bsz,)


            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            loss.backward()

            #! BEGIN successive-gradient cosine similarity addition - safe to delete to revert
            # cosine similarity (averaged over the batch) between this iteration's and the previous
            # iteration's gradient of the loss w.r.t. the transformed input image; None on the first
            # iteration, since there is no previous gradient to compare against
            grad_cosine_sim = None
            if self.config.log_progress:
                grad_x_normal = imgs.grad.detach()
                if prev_grad_x is not None:
                    grad_curr_flat = grad_x_normal.reshape(grad_x_normal.shape[0], -1)
                    grad_prev_flat = prev_grad_x.reshape(prev_grad_x.shape[0], -1)
                    grad_cosine_sim = nn.functional.cosine_similarity(
                        grad_curr_flat, grad_prev_flat, dim=1).mean().item()
                prev_grad_x = grad_x_normal
            #! END successive-gradient cosine similarity addition

            #! BEGIN alignment score addition - safe to delete to revert
            # grad_x is computed via a separate deterministic forward pass so that it is
            # consistent with the Jacobian built inside flattened_g (both use det_transformations).
            log_alignment_iter = (i + 1) % self.alignment_score_every == 0 or i == 0
            if log_alignment_iter:
                with torch.no_grad():
                    det_imgs = self.synthesize(w_batch, num_ws=self.num_ws)
                    if self.clip:
                        det_imgs = self.clip_images(det_imgs)
                    if self.det_transformations:
                        det_imgs = self.det_transformations(det_imgs)
                det_imgs = det_imgs.detach().requires_grad_(True)
                det_loss = self.attack_loss(self.target(det_imgs), targets_batch).mean()
                det_loss.backward()
                grad_x = det_imgs.grad.detach()
                self.alignment_raw_log.append({
                    'batch_idx': self.batch_idx,
                    'iteration': i,
                    'target_classes': target_classes,
                    'w': w_batch.detach().cpu(),
                    'grad_x': grad_x.cpu(),
                })
                if self.compute_alignment_score:
                    self._log_alignment_score(i, target_classes, w_batch, grad_x)
            #! END alignment score addition

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
                    self.iter_conf_log.append([self.batch_idx, i, target_classes, mean_conf.item(), grad_cosine_sim])

                if i == num_epochs - 1: #! calculate AvgTargetConf
                    for curr_target in set(targets_batch.cpu().tolist()):
                        if curr_target not in self.conf_dict:
                            self.conf_dict[curr_target] = []
                        target_indices = (targets_batch == curr_target).nonzero(as_tuple=True)[0]
                        curr_target_mean_conf = confidences[target_indices].mean().item()
                        self.conf_dict[curr_target].append(curr_target_mean_conf)

        return w_batch.detach()

    #! BEGIN alignment score addition - safe to delete this method to revert
    def _log_alignment_score(self, iteration, target_classes, w_batch, grad_x):
        scores = compute_gradient_alignment_score(
            self.synthesis, w_batch, grad_x, self.num_ws,
            transformations=self.det_transformations, chunk_size=self.alignment_score_chunk_size,
            max_samples=self.alignment_score_max_samples,
        )
        mean_score = scores.mean().item()
        for sample_idx, score in enumerate(scores.tolist()):
            self.alignment_log.append([self.batch_idx, iteration, target_classes, sample_idx, score])
        if torch.cuda.current_device() == 0:
            print(f'iteration {iteration}: \t alignment_score={mean_score:.4f}')
    #! END alignment score addition

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

    def compute_discriminator_loss(self, imgs):
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(
            -discriminator_logits).mean()
        return discriminator_loss