from __future__ import division
from __future__ import unicode_literals
################################################################################################
### Code based on: https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnv2.py ###
################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from .layers import (CondRefineBlock, RefineBlock, ResidualBlock, ncsn_conv3x3,
                     ConditionalResidualBlock, get_act)
from .normalization import get_normalization
from .sde_lib import *
from .sampling import get_sampling_fn
from tqdm import tqdm

class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the result of
        `model.parameters()`.
      decay: The exponential decay.
      use_num_updates: Whether to use number of updates when computing
        averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def update(self, parameters):
    """
    Update currently maintained parameters.

    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the same set of
        parameters used to initialize this object.
    """
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param))

  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored parameters.
    """
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']


def get_sigmas(args):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(args.sigma_max), np.log(args.sigma_min), args.num_scales))

  return sigmas


def optimization_manager(args):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=args.lr,
                  warmup=args.warmup,
                  grad_clip=args.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        optimize_fn: An optimization function.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
        A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                                  continuous=True, likelihood_weighting=likelihood_weighting)
    
    else:
        loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
        state: A dictionary of training information, containing the score model, optimizer,
        EMA status, and number of optimization steps.
        batch: A mini-batch of training/evaluation data.

        Returns:
        loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn

CondResidualBlock = ConditionalResidualBlock
conv3x3 = ncsn_conv3x3

class NCSNv2(nn.Module):
    def __init__(self, img_size, channels, args):
        super(NCSNv2, self).__init__()
        if img_size < 96:
            self.model = NCSNv2_64(args, channels=channels, image_size=img_size)
        elif 96 <= img_size <= 128:
            self.model = NCSNv2_128(args, channels=channels, image_size=img_size)
        elif 128 < img_size <= 256:
            self.model = NCSNv2_256(args, channels=channels, image_size=img_size)
        else:
            raise NotImplementedError(
                f'No network suitable for {img_size}px implemented yet.')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.img_size = img_size
        self.ema = ExponentialMovingAverage(self.model.parameters(), args.ema_decay)
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        self.num_scales = args.num_scales
        self.channels = channels

    def train_model(self, train_loader, args):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        # Build one-step training and evaluation functions
        optimize_fn = optimization_manager(args)
        continuous = args.continuous
        reduce_mean = args.reduce_mean
        likelihood_weighting = args.likelihood_weighting

        sde = VESDE(sigma_min=self.sigma_min, sigma_max=self.sigma_max, N=self.num_scales)
        sampling_eps = 1e-5

        train_step_fn = get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
        
        epoch_bar = tqdm(range(self.n_epochs), desc='Epochs', leave=True)
        self.model.train()

        for epoch in epoch_bar:
            for (x, _) in tqdm(train_loader, desc='Batches', leave=False):
                x = x.to(self.device)
                loss = train_step_fn({'model': self.model, 'optimizer': optimizer, 'ema': self.ema, 'step': epoch}, x)
                break
            self.ema.copy_to(self.model.parameters())
            epoch_bar.set_postfix(loss=loss.item())
            self.sample(args)

    @torch.no_grad()
    def sample(self, args):
        sde = VESDE(sigma_min=self.sigma_min, sigma_max=self.sigma_max, N=self.num_scales)
        sampling_eps = 1e-5
        sampling_shape = (args.batch_size, self.channels,
                          self.img_size, self.img_size)
        sampling_fn = get_sampling_fn(args, sde, sampling_shape, sampling_eps)
        sample, n = sampling_fn(self.model)
        print(sample.shape)

class NCSNv2_64(nn.Module):
  def __init__(self, args, channels=3, image_size=32):
    super().__init__()
    self.centered = args.centered
    self.norm = get_normalization(args)
    self.nf = nf = args.nf

    self.act = act = get_act(args)
    self.register_buffer('sigmas', torch.tensor(get_sigmas(args)))

    self.begin_conv = nn.Conv2d(channels, nf, 3, stride=1, padding=1)

    self.normalizer = self.norm(nf, args.num_scales)
    self.end_conv = nn.Conv2d(nf, channels, 3, stride=1, padding=1)

    self.res1 = nn.ModuleList([
      ResidualBlock(self.nf, self.nf, resample=None, act=act,
                    normalization=self.norm),
      ResidualBlock(self.nf, self.nf, resample=None, act=act,
                    normalization=self.norm)]
    )

    self.res2 = nn.ModuleList([
      ResidualBlock(self.nf, 2 * self.nf, resample='down', act=act,
                    normalization=self.norm),
      ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                    normalization=self.norm)]
    )

    self.res3 = nn.ModuleList([
      ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                    normalization=self.norm, dilation=2),
      ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                    normalization=self.norm, dilation=2)]
    )

    if image_size == 28:
      self.res4 = nn.ModuleList([
        ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                      normalization=self.norm, adjust_padding=True, dilation=4),
        ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                      normalization=self.norm, dilation=4)]
      )
    else:
      self.res4 = nn.ModuleList([
        ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                      normalization=self.norm, adjust_padding=False, dilation=4),
        ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                      normalization=self.norm, dilation=4)]
      )

    self.refine1 = RefineBlock([2 * self.nf], 2 * self.nf, act=act, start=True)
    self.refine2 = RefineBlock([2 * self.nf, 2 * self.nf], 2 * self.nf, act=act)
    self.refine3 = RefineBlock([2 * self.nf, 2 * self.nf], self.nf, act=act)
    self.refine4 = RefineBlock([self.nf, self.nf], self.nf, act=act, end=True)

  def _compute_cond_module(self, module, x):
    for m in module:
      x = m(x)
    return x

  def forward(self, x, y):
    if not self.centered:
      h = 2 * x - 1.
    else:
      h = x

    output = self.begin_conv(h)

    layer1 = self._compute_cond_module(self.res1, output)
    layer2 = self._compute_cond_module(self.res2, layer1)
    layer3 = self._compute_cond_module(self.res3, layer2)
    layer4 = self._compute_cond_module(self.res4, layer3)

    ref1 = self.refine1([layer4], layer4.shape[2:])
    ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
    ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
    output = self.refine4([layer1, ref3], layer1.shape[2:])

    output = self.normalizer(output)
    output = self.act(output)
    output = self.end_conv(output)

    used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

    output = output / used_sigmas

    return output

class NCSNv2_128(nn.Module):
  """NCSNv2 model architecture for 128px images."""
  def __init__(self, args, channels=3, image_size=128):
    super().__init__()
    self.centered = args.centered
    self.norm = get_normalization(args)
    self.nf = nf = args.nf
    self.act = act = get_act(args)
    self.register_buffer('sigmas', torch.tensor(get_sigmas(args)))

    self.begin_conv = nn.Conv2d(channels, nf, 3, stride=1, padding=1)
    self.normalizer = self.norm(nf, args.num_scales)

    self.end_conv = nn.Conv2d(nf, channels, 3, stride=1, padding=1)

    self.res1 = nn.ModuleList([
      ResidualBlock(self.nf, self.nf, resample=None, act=act,
                    normalization=self.norm),
      ResidualBlock(self.nf, self.nf, resample=None, act=act,
                    normalization=self.norm)]
    )

    self.res2 = nn.ModuleList([
      ResidualBlock(self.nf, 2 * self.nf, resample='down', act=act,
                    normalization=self.norm),
      ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                    normalization=self.norm)]
    )

    self.res3 = nn.ModuleList([
      ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                    normalization=self.norm),
      ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                    normalization=self.norm)]
    )

    self.res4 = nn.ModuleList([
      ResidualBlock(2 * self.nf, 4 * self.nf, resample='down', act=act,
                    normalization=self.norm, dilation=2),
      ResidualBlock(4 * self.nf, 4 * self.nf, resample=None, act=act,
                    normalization=self.norm, dilation=2)]
    )

    self.res5 = nn.ModuleList([
      ResidualBlock(4 * self.nf, 4 * self.nf, resample='down', act=act,
                    normalization=self.norm, dilation=4),
      ResidualBlock(4 * self.nf, 4 * self.nf, resample=None, act=act,
                    normalization=self.norm, dilation=4)]
    )

    self.refine1 = RefineBlock([4 * self.nf], 4 * self.nf, act=act, start=True)
    self.refine2 = RefineBlock([4 * self.nf, 4 * self.nf], 2 * self.nf, act=act)
    self.refine3 = RefineBlock([2 * self.nf, 2 * self.nf], 2 * self.nf, act=act)
    self.refine4 = RefineBlock([2 * self.nf, 2 * self.nf], self.nf, act=act)
    self.refine5 = RefineBlock([self.nf, self.nf], self.nf, act=act, end=True)

  def _compute_cond_module(self, module, x):
    for m in module:
      x = m(x)
    return x

  def forward(self, x, y):
    if not self.centered:
      h = 2 * x - 1.
    else:
      h = x

    output = self.begin_conv(h)

    layer1 = self._compute_cond_module(self.res1, output)
    layer2 = self._compute_cond_module(self.res2, layer1)
    layer3 = self._compute_cond_module(self.res3, layer2)
    layer4 = self._compute_cond_module(self.res4, layer3)
    layer5 = self._compute_cond_module(self.res5, layer4)

    ref1 = self.refine1([layer5], layer5.shape[2:])
    ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
    ref3 = self.refine3([layer3, ref2], layer3.shape[2:])
    ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
    output = self.refine5([layer1, ref4], layer1.shape[2:])

    output = self.normalizer(output)
    output = self.act(output)
    output = self.end_conv(output)

    used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

    output = output / used_sigmas

    return output

class NCSNv2_256(nn.Module):
  """NCSNv2 model architecture for 256px images."""
  def __init__(self, args, channels=3, image_size=256):
    super().__init__()
    self.centered = args.centered
    self.norm = get_normalization(args)
    self.nf = nf = args.nf
    self.act = act = get_act(args)
    self.register_buffer('sigmas', torch.tensor(get_sigmas(args)))

    self.begin_conv = nn.Conv2d(channels, nf, 3, stride=1, padding=1)
    self.normalizer = self.norm(nf, args.num_scales)

    self.end_conv = nn.Conv2d(nf, channels, 3, stride=1, padding=1)

    self.res1 = nn.ModuleList([
      ResidualBlock(self.nf, self.nf, resample=None, act=act,
                    normalization=self.norm),
      ResidualBlock(self.nf, self.nf, resample=None, act=act,
                    normalization=self.norm)]
    )

    self.res2 = nn.ModuleList([
      ResidualBlock(self.nf, 2 * self.nf, resample='down', act=act,
                    normalization=self.norm),
      ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                    normalization=self.norm)]
    )

    self.res3 = nn.ModuleList([
      ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                    normalization=self.norm),
      ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                    normalization=self.norm)]
    )

    self.res31 = nn.ModuleList([
      ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                    normalization=self.norm),
      ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                    normalization=self.norm)]
    )

    self.res4 = nn.ModuleList([
      ResidualBlock(2 * self.nf, 4 * self.nf, resample='down', act=act,
                    normalization=self.norm, dilation=2),
      ResidualBlock(4 * self.nf, 4 * self.nf, resample=None, act=act,
                    normalization=self.norm, dilation=2)]
    )

    self.res5 = nn.ModuleList([
      ResidualBlock(4 * self.nf, 4 * self.nf, resample='down', act=act,
                    normalization=self.norm, dilation=4),
      ResidualBlock(4 * self.nf, 4 * self.nf, resample=None, act=act,
                    normalization=self.norm, dilation=4)]
    )

    self.refine1 = RefineBlock([4 * self.nf], 4 * self.nf, act=act, start=True)
    self.refine2 = RefineBlock([4 * self.nf, 4 * self.nf], 2 * self.nf, act=act)
    self.refine3 = RefineBlock([2 * self.nf, 2 * self.nf], 2 * self.nf, act=act)
    self.refine31 = RefineBlock([2 * self.nf, 2 * self.nf], 2 * self.nf, act=act)
    self.refine4 = RefineBlock([2 * self.nf, 2 * self.nf], self.nf, act=act)
    self.refine5 = RefineBlock([self.nf, self.nf], self.nf, act=act, end=True)

  def _compute_cond_module(self, module, x):
    for m in module:
      x = m(x)
    return x

  def forward(self, x, y):
    if not self.centered:
      h = 2 * x - 1.
    else:
      h = x

    output = self.begin_conv(h)

    layer1 = self._compute_cond_module(self.res1, output)
    layer2 = self._compute_cond_module(self.res2, layer1)
    layer3 = self._compute_cond_module(self.res3, layer2)
    layer31 = self._compute_cond_module(self.res31, layer3)
    layer4 = self._compute_cond_module(self.res4, layer31)
    layer5 = self._compute_cond_module(self.res5, layer4)

    ref1 = self.refine1([layer5], layer5.shape[2:])
    ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
    ref31 = self.refine31([layer31, ref2], layer31.shape[2:])
    ref3 = self.refine3([layer3, ref31], layer3.shape[2:])
    ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
    output = self.refine5([layer1, ref4], layer1.shape[2:])

    output = self.normalizer(output)
    output = self.act(output)
    output = self.end_conv(output)

    used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

    output = output / used_sigmas

    return output