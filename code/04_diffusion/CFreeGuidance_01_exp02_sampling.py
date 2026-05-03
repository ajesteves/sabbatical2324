#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Classifier-Free Diffusion Guidance
#
# Version 1.2
#
# Source:
#
#    https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch
#
# Adapted by:
#
#    Antonio Esteves @ UMinho, April 2025
#
# TODO:
#
# * Modify `'OUR_WANDB_PROJECT_ID'`
# * Modify `'OUR_WANDB_ENTITY'`
# * In `get_celeba_config`function, modify `experiment.experiment_name`, 
#   `experiment.root_dir`, `data.data_path`, `sampling.saved_model`.
#
# ==============================================================================

import os
import ml_collections
import time
import itertools
import wandb
import random

from   datetime                     import timedelta
import numpy                        as     np
from   tqdm.notebook                import tqdm
from   abc                          import abstractmethod
from   math                         import ceil, cos, pi, log
from   PIL                          import Image
from   pathlib                      import Path
import matplotlib.pyplot            as     plt

import torch
import torch.nn                     as     nn
import torch.nn.functional          as     F
import torch.optim                  as     optim
from   torch.optim.lr_scheduler     import _LRScheduler
from   torchvision.utils            import save_image, make_grid
from   torch.nn.parallel            import DistributedDataParallel as DDP
from   torch.distributed            import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from   torch                        import Tensor
from   torchvision                  import transforms
from   torch.utils.data             import DataLoader, Dataset
from   torchvision.datasets         import CIFAR10
from   torch.utils.data.distributed import DistributedSampler
from   torchinfo                    import summary
from   torchvision                  import datasets

# ==============================================================================
# Configuration
# ==============================================================================

def get_celeba_config():
  '''
  Defines the configuration for training with CelebA dataset and CFG diffusion model.
  '''
  config = ml_collections.ConfigDict()

  config.training   = training   = ml_collections.ConfigDict()
  config.sampling   = sampling   = ml_collections.ConfigDict()
  config.data       = data       = ml_collections.ConfigDict()
  config.model      = model      = ml_collections.ConfigDict()
  config.experiment = experiment = ml_collections.ConfigDict()

  experiment.experiment_name     = "CFG_diffusion_02"
  experiment.root_dir            = "OUR_WORK_DIR_HERE"
  experiment.results_dir         = "results"
  experiment.models_dir          = "models"
  experiment.mode                = "sampling"   # "train" or "sampling" or "summary"
  experiment.saved_model         = None         # File (NOT path) with saved model to be restored or 'None' to start from scratch

  # training .....................................................
  training.batch_size            = 8         # Batch size.
  training.lr                    = 0.0002    # Learning rate.
  training.weight_decay          = 0.0001    # Weight decay.
  training.w                     = 1.8       # Classifier-free guidance strength.
  training.v                     = 0.3       # Variance of the posterior distribution.
  training.epochs                = 100       # Number of training epochs.
  training.multiplier            = 2.5       # Multiplier for learning rate warmup scheduler.
  training.threshold             = 0.1       # Threshold for classifier-free guidance.
  training.eval_interval         = 1         # Interval between successive model evaluations (in epochs).
  training.eval_images_per_class = 8         # Images to generate per class when evaluating the model.
  training.eval_images           = None      # Calculated as: eval_images_per_class * classes.
  training.checkp_interval       = 1         # Interval between successive checkpoint savings (in epochs).
  training.log_interval          = 100       # Interval between successive logs (in iterations/batches).

  # data ..........................................................
  data.data_path                 = "OUR_DATASETS_ROOT/cifar10_64x64/train" # Path to the dataset.
  data.num_workers               = 4         # Number of workers used to load training data.
  data.classes                   = 10        # Number of image classes.
  data.image_size                = 64        # Height/Width of each image.
  data.channels                  = 3         # Number of channels in each image.

  # model ..........................................................

  model.hidden_ch                = 64        # Base hidden channels in U-Net model.
  model.T                        = 1000      # Number of noise levels applied.
  model.out_ch                   = 3         # Output channels of U-Net model.
  model.ch_mult                  = [1,2,2,2] # Multiplier factor applied to 'hidden_ch' to obtain the 
                                             # Number of channels in U-Net blocks.
  model.res_blocks               = 3         # Number of residual blocks for each U-Net block.
  model.conditional_dim          = 10        # Dimension of conditional embedding.
  model.use_down_conv            = True      # Use convolution in downsample (True) or not (False).
  model.dropout_prob             = 0.2       # Dropout probability.
  model.dtype                    = torch.float32 # Data type used in U-Net and Gaussian models.

  # sampling .....................................................

  sampling.w                     = 3.0       # Classifier-free guidance strength.
  sampling.v                     = 1.0       # Variance of the posterior distribution.
  sampling.use_ddim              = True      # Use DDIM (True) or not (False).
  sampling.ddim_steps            = 100       # DDIM sampling steps.
  sampling.ddim_eta              = 0         # DDIM 'eta' applied to variance during sampling.
  sampling.ddim_select           = 'linear'  # DDIM selection strategy: 'linear' or 'quadratic'.
  sampling.batch_size            = 20        # Batch size during the sampling process.
  sampling.class_labels          = 'all'     # Labels of the images to generate: 'all', 'all_random', [list of values].
  sampling.dropout_prob          = 0.0       # Dropout probability.
  sampling.fid                   = True      # Generate samples for quantitative evaluation.
  sampling.num_images            = 5600      # Number of images to generate during sampling.
  sampling.epoch                 = 100       # Epoch ID whose model will be use during sampling.
  sampling.saved_model           = 'OUR_MODEL_NAME_HERE.pth' # File (NOT path) containing the model to use during sampling.

  training.eval_images = training.eval_images_per_class * data.classes # Images to generate when evaluating the model.

  if experiment.mode == "summary":
    config.device = 'cpu'
  else:
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  return config

def print_config(config):
  '''
  Prints the configuration.
  '''
  print('Configuration parameters:')
  for name, values in config.items():
      if isinstance(values, ml_collections.config_dict.config_dict.ConfigDict):
          print(f'{name}:')
          for key, value in values.items():
              print(f'\t{key}: {value}')
      else:
          print(f'{name}: {values}')

# Setup the environment

def setup_environment() -> ml_collections.ConfigDict:
    '''
    Read the configuration and create the necessary folders.
    '''
    config = get_celeba_config()
    print(f'Using PyTorch {torch.__version__} and {config.device} for computing')

    # Print training configuration .................................................

    print_config(config)

    # Location where we will save here the images generated during model training
    RESULTS_PATH = os.path.join(
        config.experiment.root_dir,
        config.experiment.results_dir,
        config.experiment.experiment_name
    )
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Location where the trained models will be saved
    MODELS_PATH  = os.path.join(
        config.experiment.root_dir,
        config.experiment.models_dir,
        config.experiment.experiment_name
    )
    os.makedirs(MODELS_PATH, exist_ok=True)

    GEN_IMGS_PATH = os.path.join(
        config.experiment.root_dir,
        config.experiment.results_dir,
        config.experiment.experiment_name,
        'generated_images'
    )
    os.makedirs(GEN_IMGS_PATH, exist_ok=True)

    return config

# ==============================================================================
# Gaussian Diffusion Class
# ==============================================================================

class GaussianDiffusion(nn.Module):

    def __init__(
            self,
            dtype:  torch.dtype,
            model:  nn.Module,
            betas:  np.ndarray,
            w:      float,
            v:      float,
            device: torch.device,
        ):
        super().__init__()
        self.dtype          = dtype
        self.model          = model.to(device)
        self.model.dtype    = self.dtype
        self.betas          = torch.tensor(betas, dtype=self.dtype).to(device)
        self.w              = w
        self.v              = v
        self.T              = len(betas)
        self.device         = device
        self.alphas         = 1 - self.betas
        self.log_alphas     = torch.log(self.alphas).to(self.device)

        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim = 0).to(self.device)
        self.alphas_bar     = torch.exp(self.log_alphas_bar).to(self.device)
        # self.alphas_bar   = torch.cumprod(self.alphas, dim = 0).to(self.device)

        self.log_alphas_bar_prev = F.pad(self.log_alphas_bar[:-1],[1,0],'constant', 0).to(self.device)
        self.alphas_bar_prev     = torch.exp(self.log_alphas_bar_prev).to(self.device)
        self.log_one_minus_alphas_bar_prev = torch.log(1.0 - self.alphas_bar_prev).to(self.device)
        # self.alphas_bar_prev   = F.pad(self.alphas_bar[:-1],[1,0],'constant',1).to(self.device)

        # Calculate the parameters of q(x_t|x_{t-1})

        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas     = torch.exp(self.log_sqrt_alphas).to(self.device)
        # self.sqrt_alphas   = torch.sqrt(self.alphas)

        # Calculate parameters of q(x_t|x_0)

        self.log_sqrt_alphas_bar       = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar           = torch.exp(self.log_sqrt_alphas_bar).to(self.device)
        # self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar).to(self.device)
        self.log_one_minus_alphas_bar  = torch.log(1.0 - self.alphas_bar).to(self.device)
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar).to(self.device)

        # Calculate parameters of q(x_{t-1}|x_t,x_0)

        # The log calculation is clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * torch.exp(self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.log_tilde_betas_clipped = torch.log(torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0))
        self.mu_coef_x0  = self.betas * torch.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.mu_coef_xt  = torch.exp(0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.vars        = torch.cat((self.tilde_betas[1:2],self.betas[1:]), 0)
        self.coef1       = torch.exp(-self.log_sqrt_alphas)
        self.coef2       = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar
        # Calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar   = torch.exp(-self.log_sqrt_alphas_bar)
        # self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar)
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)

    @staticmethod
    def _extract(coef: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Arguments:

        * coef :    an array
        * t :       timestep
        * x_shape : the shape of tensor x that has K dims(the value of first dim is batch size)

        Returns:
           A tensor of shape [batch_size,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]

        neo_shape    = torch.ones_like(torch.tensor(x_shape)).to(t.device)
        neo_shape[0] = x_shape[0]
        neo_shape    = neo_shape.tolist()
        chosen       = coef[t]
        chosen       = chosen.to(t.device)
        return chosen.reshape(neo_shape)

    def q_mean_variance(self, x_0:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the parameters of q(x_t|x_0).
        """
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var  = self._extract(1.0 - self.sqrt_alphas_bar, t, x_0.shape)
        return mean, var

    def q_sample(self, x_0:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t|x_0)
        """
        eps = torch.randn_like(x_0, requires_grad=False, device=self.device)
        return self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 \
            + self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps, eps

    def q_posterior_mean_variance(
        self, 
        x_0:torch.Tensor, 
        x_t:torch.Tensor, 
        t:torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the parameters of q(x_{t-1}|x_t,x_0).
        """
        posterior_mean        = self._extract(self.mu_coef_x0, t, x_0.shape) * x_0 \
                              + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        posterior_var_max     = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(self.log_tilde_betas_clipped, t, x_t.shape)
        log_posterior_var_max = self._extract(torch.log(self.betas), t, x_t.shape)
        log_posterior_var     = self.v * log_posterior_var_max + (1 - self.v) * log_posterior_var_min
        neo_posterior_var     = torch.exp(log_posterior_var)

        return posterior_mean, posterior_var_max, neo_posterior_var

    def p_mean_variance(
        self, 
        x_t:torch.Tensor, 
        t:torch.Tensor, 
        **model_kwargs
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the parameters of p_{theta}(x_{t-1}|x_t).
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape           = model_kwargs['cemb'].shape
        pred_eps_cond        = self.model(x_t, t, **model_kwargs)
        model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        pred_eps_uncond      = self.model(x_t, t, **model_kwargs)
        pred_eps             = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond

        assert torch.isnan(x_t).int().sum() == 0,      f"[ERROR] nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0,        f"[ERROR] nan in tensor t when   t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"[ERROR] nan in tensor pred_eps when t = {t[0]}"

        p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps)
        p_var  = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        return p_mean, p_var

    def _predict_x0_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.sqrt_recip_alphas_bar, t = t, x_shape = x_t.shape) \
            * x_t - self._extract(coef = self.sqrt_one_minus_alphas_bar, t = t, x_shape = x_t.shape) * eps

    def _predict_xt_prev_mean_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.coef1, t = t, x_shape = x_t.shape) * x_t - \
            self._extract(coef = self.coef2, t = t, x_shape = x_t.shape) * eps

    def p_sample(self, x_t:torch.Tensor, t:torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        Sample x_{t-1} from p_{theta}(x_{t-1}|x_t).
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"[ERROR] size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t , t, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"[ERROR] nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0,  f"[ERROR] nan in tensor var  when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise

    def sample(self, shape: tuple, **model_kwargs) -> torch.Tensor:
        """
        Sample images from p_{theta}.
        """
        local_rank = get_rank()
        if local_rank == 0:
            print('[INFO] start sampling ...')
        if model_kwargs == None:
            model_kwargs = {}
        x_t   = torch.randn(shape, device = self.device)
        tlist = torch.ones([x_t.shape[0]], device = self.device) * self.T
        for _ in tqdm(range(self.T),dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('[INFO] ... end sampling')
        return x_t

    def ddim_p_mean_variance(
        self, 
        x_t:torch.Tensor, 
        t:torch.Tensor, 
        prevt:torch.Tensor, 
        eta:float, 
        **model_kwargs
         -> torch.Tensor:
        """
        Calculate the parameters of p_{theta}(x_{t-1}|x_t) used by DDIM.
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape           = model_kwargs['cemb'].shape
        pred_eps_cond        = self.model(x_t, t, **model_kwargs)
        model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        pred_eps_uncond      = self.model(x_t, t, **model_kwargs)
        pred_eps             = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond

        assert torch.isnan(x_t).int().sum() == 0,      f"[ERROR] nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0,        f"[ERROR] nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"[ERROR] nan in tensor pred_eps when t = {t[0]}"

        alphas_bar_t    = self._extract(coef = self.alphas_bar, t = t, x_shape = x_t.shape)
        alphas_bar_prev = self._extract(coef = self.alphas_bar_prev, t = prevt + 1, x_shape = x_t.shape)
        sigma           = eta * torch.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar_t) * (1 - alphas_bar_t / alphas_bar_prev))
        p_var           = sigma ** 2
        coef_eps        = 1 - alphas_bar_prev - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps        = torch.sqrt(coef_eps)
        p_mean          = torch.sqrt(alphas_bar_prev) * (x_t - torch.sqrt(1 - alphas_bar_t) * pred_eps) / \
                          torch.sqrt(alphas_bar_t) + coef_eps * pred_eps
        return p_mean, p_var

    def ddim_p_sample(
        self, 
        x_t:torch.Tensor, 
        t:torch.Tensor, 
        prevt:torch.Tensor, 
        eta:float, 
        **model_kwargs
        ) -> torch.Tensor:
        """
        Sample x_{t-1} from p_{theta}(x_{t-1}|x_t) using DDIM.
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"[ERROR] size of t is not batch size {B}"
        mean, var = self.ddim_p_mean_variance(x_t , t.type(dtype=torch.long), prevt.type(dtype=torch.long), eta, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"[ERROR] nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0,  f"[ERROR] nan in tensor var  when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise

    def ddim_sample(self, shape: tuple, num_steps: int, eta: float, select: str, **model_kwargs) -> torch.Tensor:
        """
        Sample images from p_{theta} using DDIM.
        """
        local_rank = get_rank()
        if local_rank == 0:
            print('[INFO] start sampling with ddim ...')
        if model_kwargs == None:
            model_kwargs = {}

        # a subsequence of range(0,1000)
        if select == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        elif select == 'quadratic':
            tseq = list((np.linspace(0, np.sqrt(self.T), num_steps-1)**2).astype(int))
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(f'[ERROR] There is no ddim discretization method called "{select}"')

        x_t   = torch.randn(shape, device = self.device)
        tlist = torch.zeros([x_t.shape[0]], device = self.device)
        for i in tqdm(range(num_steps),dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            with torch.no_grad():
                tlist = tlist * 0 + tseq[-1-i]
                if i != num_steps - 1:
                    prevt = torch.ones_like(tlist, device = self.device) * tseq[-2-i]
                else:
                    prevt = - torch.ones_like(tlist, device = self.device) 
                x_t = self.ddim_p_sample(x_t, tlist, prevt, eta, **model_kwargs)
                torch.cuda.empty_cache()
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('[INFO] ... end sampling process with ddim')
        return x_t

    def trainloss(self, x_0:torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        Calculate the loss used by denoising diffusion probabilistic model.
        """
        if model_kwargs == None:
            model_kwargs = {}
        t        = torch.randint(self.T, size = (x_0.shape[0],), device=self.device)
        x_t, eps = self.q_sample(x_0, t)
        pred_eps = self.model(x_t, t, **model_kwargs)
        loss     = F.mse_loss(pred_eps, eps, reduction='mean')
        return loss

# ==============================================================================
# Learning Rate Scheduler
# ==============================================================================

class GradualWarmupScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            multiplier,
            warm_epoch,
            after_scheduler = None,
            last_epoch      = None,
        ):
        self.multiplier      = multiplier
        self.total_epoch     = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished        = False
        self.last_epoch      = last_epoch
        self.base_lrs        = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def state_dict(self):
        warmdict = {key:value for key, value in self.__dict__.items() if (key != 'optimizer' and key != 'after_scheduler')}
        cosdict  = {key:value for key, value in self.after_scheduler.__dict__.items() if key != 'optimizer'}
        return {'warmup':warmdict, 'afterscheduler':cosdict}

    def load_state_dict(self, state_dict: dict):
        self.after_scheduler.__dict__.update(state_dict['afterscheduler'])
        self.__dict__.update(state_dict['warmup'])

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

# ==============================================================================
# Noise Variance Schedule
# ==============================================================================

def get_named_beta_schedule(schedule_name='linear', num_diffusion_timesteps=1000) -> np.ndarray:
    """
    Get a predefined 'beta' schedule given the schedule name.

    The 'beta' schedule library consists of 'beta' schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al,
        # extended to work with any number of diffusion steps.
        scale      = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end   = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return  betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: cos((t + 0.008) / 1.008 * pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"[ERROR] unknown beta schedule '{schedule_name}'")

def betas_for_alpha_bar(num_diffusion_timesteps:int, alpha_bar, max_beta=0.999) -> np.ndarray:
    """
    Create a 'beta' schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time t, for t in [0,1].

    Arguments:
    * num_diffusion_timesteps: the number of betas to produce.
    * alpha_bar: a lambda that takes an argument t from 0 to 1 and
                 produces the cumulative product of (1-beta) up to that
                 part of the diffusion process.
    * max_beta:  the maximum beta to use; use values lower than 1 to
                 prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# ==============================================================================
# U-Net Model
# ==============================================================================

def timestep_embedding(
        timesteps:  torch.Tensor,
        dim:        int,
        max_period: int  = 10000,
    ) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Arguments:
    * timesteps:  A 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
    * dim:        The dimension of the output.
    * max_period: Controls the minimum frequency of the embeddings.
    Returns:
      An [N x dim] tensor of positional embeddings.
    """
    half  = dim // 2
    freqs = torch.exp(
        -log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args      = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Upsample(nn.Module):
    """
    An upsampling layer.
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        x = F.interpolate(x, scale_factor = 2, mode = "nearest")
        output = self.layer(x)
        return output


class Downsample(nn.Module):
    """
    A downsampling layer.
    """
    def __init__(self, in_ch: int, out_ch: int, use_conv: bool):
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        if use_conv:
            self.layer = nn.Conv2d(
                self.in_ch,
                self.out_ch,
                kernel_size = 3,
                stride      = 2,
                padding     = 1,
            )
        else:
            self.layer = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        return self.layer(x)


class EmbedBlock(nn.Module):
    """
    Abstract class.
    """
    @abstractmethod
    def forward(self, x, temb, cemb):
        """
        Abstract method.
        """


class EmbedSequential(nn.Sequential, EmbedBlock):

    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb)
            else:
                x = layer(x)
        return x


class ResBlock(EmbedBlock):
    '''
    Residual block used by U-Net.
    '''
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, tdim:int, cdim:int, dropout_prob:float):
        super().__init__()
        self.in_ch        = in_ch
        self.out_ch       = out_ch
        self.tdim         = tdim
        self.cdim         = cdim
        self.dropout_prob = dropout_prob

        self.block_1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        self.cemb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cdim, out_ch),
        )

        self.block_2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(p = self.dropout_prob),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
        )
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.residual = nn.Identity()

    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        latent  = self.block_1(x)
        latent += self.temb_proj(temb)[:, :, None, None]
        latent += self.cemb_proj(cemb)[:, :, None, None]
        latent  = self.block_2(latent)
        latent += self.residual(x)
        return latent


class AttnBlock(nn.Module):
    '''
    Attention block used by U-Net.
    '''
    def __init__(self, in_ch:int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q     = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj_k     = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj_v     = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj       = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class Unet(nn.Module):
    '''
    The U-Net module.
    '''
    def __init__(
            self,
            in_ch          = 3,
            hidden_ch      = 64,
            out_ch         = 3,
            ch_mult        = [1,2,4,8],
            num_res_blocks = 2,
            cdim           = 10,
            use_conv       = True,
            dropout_prob   = 0,
            dtype          = torch.float32,
        ):
        super().__init__()
        self.in_ch          = in_ch
        self.hidden_ch      = hidden_ch
        self.out_ch         = out_ch
        self.ch_mult        = ch_mult
        self.num_res_blocks = num_res_blocks
        self.cdim           = cdim
        self.use_conv       = use_conv
        self.dropout_prob   = dropout_prob
        self.dtype          = dtype
        tdim                = hidden_ch * 4

        self.temb_layer = nn.Sequential(
            nn.Linear(hidden_ch, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.cemb_layer = nn.Sequential(
            nn.Linear(self.cdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.downblocks = nn.ModuleList([
            EmbedSequential(nn.Conv2d(in_ch, self.hidden_ch, 3, padding=1))
        ])
        now_ch = self.ch_mult[0] * self.hidden_ch
        chs    = [now_ch]
        for i, mul in enumerate(self.ch_mult):
            nxt_ch = mul * self.hidden_ch
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(now_ch, nxt_ch, tdim, tdim, self.dropout_prob),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mult) - 1:
                self.downblocks.append(EmbedSequential(Downsample(now_ch, now_ch, self.use_conv)))
                chs.append(now_ch)
        self.middleblocks = EmbedSequential(
            ResBlock(now_ch, now_ch, tdim, tdim, self.dropout_prob),
            AttnBlock(now_ch),
            ResBlock(now_ch, now_ch, tdim, tdim, self.dropout_prob)
        )
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mult))[::-1]:
            nxt_ch = mul * self.hidden_ch
            for j in range(num_res_blocks + 1):
                layers = [
                    ResBlock(now_ch+chs.pop(), nxt_ch, tdim, tdim, self.dropout_prob),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential(*layers))
        self.out = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, self.out_ch, 3, stride = 1, padding = 1)
        )

    def forward(self, x:torch.Tensor, t:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:

        temb = self.temb_layer(timestep_embedding(t, self.hidden_ch))
        cemb = self.cemb_layer(cemb)
        hs   = []
        h    = x.type(self.dtype)
        for block in self.downblocks:
            h = block(h, temb, cemb)
            hs.append(h)
        h    = self.middleblocks(h, temb, cemb)
        for block in self.upblocks:
            h = torch.cat([h, hs.pop()], dim = 1)
            h = block(h, temb, cemb)
        h    = h.type(self.dtype)
        return self.out(h)

# ==============================================================================
# Timestep Embedding Layer
# ==============================================================================

class ConditionalEmbedding(nn.Module):
    '''
    Embedding layer for conditioning timestep input 't'.
    '''
    def __init__(self, num_labels:int, d_model:int, dim:int):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(t)
        return emb


# ==============================================================================
# Create a custom Dataset from the images in a folder.
# NOTE: WITH CIFAR10 THIS CLASS IS NOT USED.
# ==============================================================================

class CustomDataSet(Dataset):

    def __init__(self, root_dir, transform):
        self.root_dir     = root_dir
        self.transform    = transform
        self.all_images   = os.listdir(root_dir)
        self.total_images = natsorted(self.all_images)

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        img_loc      = os.path.join(self.root_dir, self.total_images[idx])
        image        = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


# ==============================================================================
# Create a DataLoader
# 
# Define the transformations that will be applied to the images:
# 
# * convert the images to tensors
# * crop the images
# * resize the images
# * normalize the images.
# 
# Instantiate a Custom Dataset
# Create a training DataLoader
# ==============================================================================

class Data_Loader():
    '''
    DataLoader class that works with LSUN and CelebA datasets.
    '''
    def __init__(
            self,
            dataset,
            images_path,
            image_size,
            crop_size,
            resize,
            normalize,
            centercrop,
            batch_size,
            shuffle = True,
        ):
        self.dataset_name = dataset
        self.path         = images_path
        self.image_size   = image_size
        self.crop_size    = crop_size
        self.resize       = resize
        self.normalize    = normalize
        self.centercrop   = centercrop
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.length       = 0 

    def transform(self):

        options = []

        options.append(transforms.ToTensor())
        print('[INFO] Added ToTensor transform ...')

        if self.centercrop:
            offset_height = (218 - self.crop_size) // 2
            offset_width  = (178 - self.crop_size) // 2
            crop = lambda x: x[:, offset_height:offset_height + self.crop_size, offset_width:offset_width + self.crop_size]
            options.append(transforms.Lambda(crop))
            print('[INFO] Added Crop transform ...') 

        if self.resize:
            options.append(transforms.Resize((self.image_size, self.image_size)))
            print('[INFO] Added Resize transform ...') 

        if self.normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            print('[INFO] Added Normalize transform ...')

        transform = transforms.Compose(options)

        return transform

    def load_cifar10(self, classes='church_outdoor_train'):
        transforms = self.transform()
        dataset    = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform()
        dataset    = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform()
        dataset    = CustomDataSet(
            root_dir  = self.path,
            transform = transforms,
        )
        return dataset

    def loader(self):
        if self.dataset_name == 'lsun':
            self.dataset = self.load_lsun()
        elif self.dataset_name == 'CELEBA balanced':
            self.dataset = self.load_celeb()
        size        = len(self.dataset)
        self.length = int(size / self.batch_size)
        print(f'[INFO] Dataset length:    {size}')
        print(f'[INFO] DataLoader length: {self.length}')

        loader = torch.utils.data.DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            shuffle     = self.shuffle,
            num_workers = 2,
            drop_last   = True,
        )
        return loader


def check_dataloader_v2(
        dataset,
        images_path,
        image_size,
        crop_size,
        resize,
        normalize,
        centercrop,
        batch_size,
        shuffle=True
    ):
    NR, NC    = 3, 3

    DLoader = Data_Loader(
        dataset     = dataset,
        images_path = images_path,
        image_size  = image_size,
        crop_size   = crop_size,
        resize      = resize,
        normalize   = normalize,
        centercrop  = centercrop,
        batch_size  = batch_size,
        shuffle     = shuffle,
    )

    loader = DLoader.loader()
    imgs   = next(iter(loader))

    print(f'[INFO] Batch of images shape: {imgs.shape}')   # BS, Ch, H, W

    if NR*NC > imgs.shape[0]:
        NR = 2
        if NR*NC > imgs.shape[0]:
            NR = 1
            if NR*NC > imgs.shape[0]:
                NC = 2

    _, ax    = plt.subplots(NR, NC, figsize=(3*NC,3*NR))
    plt.suptitle(
        f'Some real images of {dataset} dataset',
        fontsize   = 15,
        fontweight = 'bold',
    )

    index = 0
    for r in range(NR):
        for c in range(NC):
            index += 1
            if NR==1:
                #ax[c].imshow(imgs[index].permute(1,2,0)+1)/2)
                ax[c].imshow(imgs[index].permute(1,2,0))
            else:
                #ax[r][c].imshow((imgs[index].permute(1,2,0)+1)/2)
                ax[r][c].imshow(imgs[index].permute(1,2,0))


def print_random_image(dataset_path: Path):
    '''
    Display a randomly selected image from dataset.
    '''
    # Setup path to data folder
    image_path  = Path(dataset_path)

    # 1. Get all image paths
    image_path_list = list(image_path.glob("*/*.png"))

    # 2. Get random image path
    random_image_path = random.choice(image_path_list)

    # 3. Get image class from path name (the image class is 
    #    the name of the directory where the image is stored)
    image_class = random_image_path.parent.stem

    # 4. Open image
    img = Image.open(random_image_path)

    # 5. Print metadata
    print(f"[INFO] random image path: {random_image_path}")
    print(f"[INFO] image class:  {image_class}")
    print(f"[INFO] image height: {img.height}")
    print(f"[INFO] image width:  {img.width}")
    img


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    '''
    Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    '''
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


def load_data(
    data_path:   str,
    batch_size:  int,
    num_workers: int,
    shuffle:     bool = True,
    ) -> tuple[DataLoader, DistributedSampler]:

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dset = datasets.ImageFolder(
        root             = data_path,      # target folder of images
        transform        = data_transform, # transforms to perform on data (images)
        target_transform = None,           # no transformation is applied to labels
    )

    class_names = dset.classes
    print(f'[INFO] training dataset classes: {class_names}')

    # Check the lengths
    print(f'[INFO] training dataset length: {len(dset)}')

    sampler = DistributedSampler(dset, shuffle=shuffle) # use 'shuffle=True' in DistributedSample, not in DataLoader

    # Turn train Dataset into a DataLoader
    trainloader = DataLoader(
        dataset     = dset,
        batch_size  = batch_size,  # how many samples per batch?
        num_workers = num_workers, # how many subprocesses to use for data loading? (higher = more)
        shuffle     = False,       # use 'shuffle=True' in DistributedSample, not in DataLoader
        drop_last   = True,        # discard last batch
        sampler     = sampler,
    )

    return trainloader, sampler


def undo_normalize(data:Tensor) -> Tensor:
    return data / 2 + 0.5


def check_dataloader(
        dataset_name,
        dataloader,
    ):

    NR, NC         = 3, 3
    imgs, labels   = next(iter(dataloader))

    print(f'[INFO] batch of images shape: {imgs.shape}')   # BS, Ch, H, W
    print(f"[INFO] batch of labels shape: {labels.shape}")

    imgs = undo_normalize(imgs)

    if NR*NC > imgs.shape[0]:
        NR = 2
        if NR*NC > imgs.shape[0]:
            NR = 1
            if NR*NC > imgs.shape[0]:
                NC = 2

    _, ax    = plt.subplots(NR, NC, figsize=(3*NC,3*NR))
    plt.suptitle(
        f'Some real images of {dataset_name} dataset',
        fontsize   = 15,
        fontweight = 'bold',
    )

    index = 0
    for r in range(NR):
        for c in range(NC):
            index += 1
            if NR==1:
                #ax[c].imshow(imgs[index].permute(1,2,0)+1)/2)
                ax[c].imshow(imgs[index].permute(1,2,0))
            else:
                #ax[r][c].imshow((imgs[index].permute(1,2,0)+1)/2)
                ax[r][c].imshow(imgs[index].permute(1,2,0))

def time_format(seconds: int) -> str:
    '''
    Converts a time in seconds to days:hours:minutes:seconds.
    '''
    if seconds is not None:
        seconds = int(seconds)
        d = seconds // (3600 * 24)
        h = seconds // 3600 % 24
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        if d > 0:
            return '{:02d}D {:02d}H {:02d}m {:02d}s'.format(d, h, m, s)
        elif h > 0:
            return '{:02d}H {:02d}m {:02d}s'.format(h, m, s)
        elif m > 0:
            return '{:02d}m {:02d}s'.format(m, s)
        elif s > 0:
            return '{:02d}s'.format(s)
    return '-'

# ==============================================================================
# Trainer Class
# ==============================================================================

class Trainer():
    '''
    Class to manage training and evaluation of the model.
    '''
    # =========================================================================
    def __init__(
        self,
        config,
        ):
        self.config          = config
        self.world_size      = int(os.environ["WORLD_SIZE"])
        self.local_rank      = int(os.environ["LOCAL_RANK"])
        self.unet            = None
        self.cemblayer       = None
        self.diffusion       = None
        self.optimizer       = None
        self.cosineScheduler = None
        self.warmUpScheduler = None
        self.epoch           = 0
        self.step            = 0

    # ==============================================================================
    def train(self):

        assert self.config.training.eval_images % (self.world_size * self.config.data.classes) == 0 , \
            '[ERROR] please correct "training.eval_images" parameter!'

        # Initialize torch distributed training ................................

        # 'torchrun' assigns RANK and WORLD_SIZE automatically, among other environment variables.
        # Set device ID using LOCAL_RANK provided by 'torchrun'.
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend="nccl")

        # Get local rank for each process
        self.local_rank = get_rank()

        # Set computing device

        self.config.device = torch.device("cuda", self.local_rank)

        # Create a data loader .................................................

        dataloader, sampler = load_data(
            data_path   = self.config.data.data_path,
            batch_size  = self.config.training.batch_size,
            num_workers = self.config.data.num_workers,
            shuffle     = True,
        )

        if self.local_rank==0:
            check_dataloader("CIFAR10 64x64", dataloader)

        # Initialize the models ................................................
        self.unet = Unet(
            in_ch          = self.config.data.channels,
            hidden_ch      = self.config.model.hidden_ch,
            out_ch         = self.config.model.out_ch,
            ch_mult        = self.config.model.ch_mult,
            num_res_blocks = self.config.model.res_blocks,
            cdim           = self.config.model.conditional_dim,
            use_conv       = self.config.model.use_down_conv,
            dropout_prob   = self.config.model.dropout_prob,
            dtype          = self.config.model.dtype,
        )

        self.cemblayer = ConditionalEmbedding(
            self.config.data.classes,
            self.config.model.conditional_dim,
            self.config.model.conditional_dim,
        ).to(self.config.device)

        # Get the beta schedule to use in the diffusion process
        betas = get_named_beta_schedule(num_diffusion_timesteps = self.config.model.T)

        self.diffusion = GaussianDiffusion(
            dtype  = self.config.model.dtype,
            model  = self.unet,
            betas  = betas,
            w      = self.config.training.w,
            v      = self.config.training.v,
            device = self.config.device,
        )

        # Load a saved model ...................................................

        checkpoint = self.restore_checkpoint(self.config.experiment.saved_model)

        # Select the optimizer and the learning rate schedulers ................
        self.optimizer = torch.optim.AdamW(
            itertools.chain(
                self.diffusion.model.parameters(),
                self.cemblayer.parameters()
            ),
            lr           = self.config.training.lr,
            weight_decay = self.config.training.weight_decay,
        )

        self.cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer  = self.optimizer,
            T_max      = self.config.training.epochs,
            eta_min    = 0,
            last_epoch = -1,
        )

        self.warmUpScheduler = GradualWarmupScheduler(
            optimizer       = self.optimizer,
            multiplier      = self.config.training.multiplier,
            warm_epoch      = self.config.training.epochs // 10,
            after_scheduler = self.cosineScheduler,
            last_epoch      = self.epoch,
        )

        if checkpoint is not None:
            # load saved optimizer and scheduler states ........................
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.warmUpScheduler.load_state_dict(checkpoint['scheduler'])

        # Wrap the model for Distributed Data Parallel (DDP) training ..........

        self.diffusion.model = DDP(
            self.diffusion.model,
            device_ids    = [self.local_rank],
            output_device = self.local_rank,
        )

        self.cemblayer = DDP(
            self.cemblayer,
            device_ids    = [self.local_rank],
            output_device = self.local_rank,
        )

        # Training loop ........................................................

        mean_loss = {'value': 0.0, 'cnt': 0}

        for epc in range(self.epoch, self.config.training.epochs):

            ts  = time.time()

            # Put the model in training mode
            self.diffusion.model.train()
            self.cemblayer.train()
            sampler.set_epoch(epc)

            # Train the model for an epoch .....................................

            # Only process with rank=0 prints the progress bar
            with tqdm(dataloader, dynamic_ncols=True, disable=(self.local_rank % self.world_size != 0)) as tqdmloader:
                for img, label in tqdmloader:

                    b       = img.shape[0]
                    self.optimizer.zero_grad()
                    x_0     = img.to(self.config.device)
                    label   = label.to(self.config.device)
                    cemb    = self.cemblayer(label)
                    cemb[np.where(np.random.rand(b)<self.config.training.threshold)] = 0
                    loss    = self.diffusion.trainloss(x_0, cemb = cemb)
                    loss.backward()
                    self.optimizer.step()

                    img_shape = [x_0.shape[1], x_0.shape[2], x_0.shape[3]]
                    tqdmloader.set_postfix(
                        ordered_dict = {
                            "epoch":            epc + 1,
                            "loss":             loss.item(),
                            "LR":               self.optimizer.state_dict()['param_groups'][0]["lr"],
                            "device batch":     x_0.shape[0],
                            "img shape":        img_shape
                        }
                    )

                    # Update the mean loss
                    mean_loss['value'] = (mean_loss['cnt']*mean_loss['value'])/(mean_loss['cnt']+1) + \
                        loss.item()/(mean_loss['cnt']+1)
                    mean_loss['cnt'] += 1

                    # Log the loss, step and epoch to Weights and Biases .......
                    if (self.step+1) % self.config.training.log_interval == 0 and self.local_rank == 0:
                        try:
                            # Log metrics to Weights and Biases 
                            wandb.log(
                                {
                                "loss":  mean_loss['value'],
                                "step":  self.step+1,
                                "epoch": self.epoch+1
                                }
                            )
                        except Exception as e:
                            print(f'[ERROR] (#1) An exception of type {type(e).__name__} occurred. Arguments:\n{ex.args!r}')

                        mean_loss['value'] = 0.0
                        mean_loss['cnt']   = 0

                    self.step += 1

            # End of the epoch .................................................

            te        = time.time()
            texec_sec = te - ts
            texec_str = time_format(texec_sec)
            print(f'\nEpoch training time: {texec_str}')

            if (self.local_rank == 0):
                try:
                    wandb.log(
                        {
                        "epoch_training_time_sec": texec_sec,
                        }
                    )
                except Exception as ex:
                    print(f'[ERROR] An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args!r}')

            # Update the learning rate
            self.warmUpScheduler.step()

            # Save the model checkpoint ........................................

            if (self.local_rank == 0) and ( (epc + 1) % self.config.training.checkp_interval==0 \
                or (epc+1)==self.config.training.epochs ):
                model_file = self.set_file_name(
                    self.config.experiment.models_dir,
                    postfix   = None,
                    extension = 'pth',
                )
                self.save_checkpoint(model_file)

            # Evaluate the model ...............................................

            if (epc + 1) % self.config.training.eval_interval == 0:
                self.diffusion.model.eval()
                self.cemblayer.eval()

                # Generate samples .............................................

                # The model generates 80 images (8 per row) each time.
                # Images in the same row will belong to the same class.
                all_samples       = []
                each_device_batch = self.config.training.eval_images // self.world_size
                with torch.no_grad():
                    label    = torch.ones(self.config.data.classes, each_device_batch // self.config.data.classes).type(torch.long) \
                    * torch.arange(start = 0, end = self.config.data.classes).reshape(-1, 1)
                    label    = label.reshape(-1, 1).squeeze()
                    label    = label.to(self.config.device)
                    cemb     = self.cemblayer(label)
                    genshape = (
                        each_device_batch,
                         self.config.data.channels,
                         self.config.data.image_size,
                         self.config.data.image_size
                    )
                    if self.config.sampling.use_ddim:
                        generated = self.diffusion.ddim_sample(
                            genshape,
                            self.config.sampling.ddim_steps,
                            self.config.sampling.ddim_eta,
                            self.config.sampling.ddim_select,
                            cemb = cemb,
                        )
                    else:
                        generated = self.diffusion.sample(genshape, cemb = cemb)

                    img = undo_normalize(generated)
                    img = img.reshape(
                        self.config.data.classes, each_device_batch // self.config.data.classes, 
                        self.config.data.channels,
                        self.config.data.image_size,
                        self.config.data.image_size
                    ).contiguous()

                    gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
                    all_gather(gathered_samples, img)
                    all_samples.extend([img for img in gathered_samples])

                    # each_device_batch: 80/world_size
                    # generated: [80/world_size, CHANNELS, IMG_SIZE, IMG_SIZE]
                    # img: [CLASSES, 8, CHANNELS, IMG_SIZE, IMG_SIZE]
                    # gathered_samples: List of length 'world_size' where each element is [CLASSES, 8, CHANNELS, IMG_SIZE, IMG_SIZE]
                    # all_samples: List of length 'world_size' where each element is [CLASSES, 8, CHANNELS, IMG_SIZE, IMG_SIZE]

                    samples = torch.concat(all_samples, dim = 1).reshape(
                        self.config.training.eval_images,
                        self.config.data.channels,
                        self.config.data.image_size,
                        self.config.data.image_size
                    )

                    # Save the generated images to file ........................
                    if self.local_rank == 0:
                        imgs_file = self.set_file_name(
                            self.config.experiment.results_dir,
                            postfix   = "generated",
                            extension = 'png',
                        )
                        grid = make_grid(
                            samples,
                            nrow = self.config.training.eval_images_per_class,
                        )
                        save_image(grid, imgs_file)

            torch.cuda.empty_cache()
            self.epoch += 1

        # Delete the distributed training processes ............................
        destroy_process_group()


    # ===================================================================================
    @torch.no_grad()
    def sample(self):

        if self.config.sampling.class_labels == 'all' or self.config.sampling.class_labels == 'all_random':
            num_sampling_classes = self.config.data.classes
        else:
            if isinstance(self.config.sampling.class_labels, list):
                num_sampling_classes = len(self.config.sampling.class_labels)
            else:
                print(f'[ERROR] invalid value in parameter "sampling.class_labels"')
                return

        assert self.config.sampling.batch_size % (self.world_size * num_sampling_classes) == 0 , \
            '[ERROR] please correct "sampling.batch_size" parameter!'

        # Initialize the distributed training processses .......................

        # 'torchrun' assigns RANK and WORLD_SIZE automatically, among other environment variables.
        # Set device ID using LOCAL_RANK provided by 'torchrun'.
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend="nccl")

        # Get local rank for each process
        self.local_rank = get_rank()

        # Set computing device

        self.config.device = torch.device("cuda", self.local_rank)

        # Instantiate the models ...............................................

        self.unet = Unet(
            in_ch          = self.config.data.channels,
            hidden_ch      = self.config.model.hidden_ch,
            out_ch         = self.config.model.out_ch,
            ch_mult        = self.config.model.ch_mult,
            num_res_blocks = self.config.model.res_blocks,
            cdim           = self.config.model.conditional_dim,
            use_conv       = self.config.model.use_down_conv,
            dropout_prob   = self.config.sampling.dropout_prob,
            dtype          = self.config.model.dtype,
        ).to(self.config.device)

        self.cemblayer  = ConditionalEmbedding(
            self.config.data.classes,
            self.config.model.conditional_dim,
            self.config.model.conditional_dim,
        ).to(self.config.device)

        # Get the beta schedule to use in the diffusion process ................

        betas = get_named_beta_schedule(num_diffusion_timesteps = self.config.model.T)
        self.diffusion = GaussianDiffusion(
            dtype  = self.config.model.dtype,
            model  = self.unet,
            betas  = betas,
            w      = self.config.sampling.w,
            v      = self.config.sampling.v,
            device = self.config.device,
        )

        # Load the model from file .............................................

        checkpoint = self.restore_checkpoint(self.config.sampling.saved_model)

        # Wrap the model for using Distributed Data Parallel (DDP) .............

        self.diffusion.model = DDP(
            self.diffusion.model,
            device_ids    = [self.local_rank],
            output_device = self.local_rank,
        )
        self.cemblayer = DDP(
            self.cemblayer,
            device_ids    = [self.local_rank],
            output_device = self.local_rank,
        )

        # Put the models in evaluation mode
        self.diffusion.model.eval()
        self.cemblayer.eval()

        # Number of sampling batches
        if self.config.sampling.fid:
            numloop = ceil(self.config.sampling.num_images  / self.config.sampling.batch_size)
        else:
            numloop = 1

        each_device_batch = self.config.sampling.batch_size // self.world_size

        # Set the class labels of the images to sample (all classes in this case)
        if self.config.sampling.class_labels == 'all':
            label = torch.ones(
                self.config.data.classes,
                each_device_batch // self.config.data.classes
            ).type(torch.long) * torch.arange(
                start = 0,
                end   = self.config.data.classes
            ).reshape(-1,1)
            label = label.reshape(-1, 1).squeeze()
            label = label.to(self.config.device)

        elif self.config.sampling.class_labels == 'all_random':
            label = torch.randint(
                low    = 0,
                high   = self.config.data.classes,
                size   = (each_device_batch,),
                device = self.config.device,
            )

        elif isinstance(self.config.sampling.class_labels, list):
            label = torch.tensor(
                self.config.sampling.class_labels,
                dtype  = torch.long,
                device = self.config.device,
            )

        # Get label embeddings
        cemb        = self.cemblayer(label)
        genshape    = (
            each_device_batch,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size
        )
        all_samples = []
        if self.local_rank == 0:
            print(numloop)

        # Generate images with the diffusion model .............................
        for it in range(numloop):

            if self.local_rank == 0:
                print(f'[INFO] sampling iteration {it+1} of {numloop} | labels: {label}')

            if self.config.sampling.use_ddim:
                generated = self.diffusion.ddim_sample(
                    genshape,
                    self.config.sampling.ddim_steps,
                    self.config.sampling.ddim_eta,
                    self.config.sampling.ddim_select,
                    cemb = cemb,
                )
            else:
                generated = self.diffusion.sample(genshape, cemb = cemb)

            # Transform samples into images
            img = undo_normalize(generated)
            img = img.reshape(
                self.config.data.classes,
                each_device_batch // self.config.data.classes,
                self.config.data.channels,
                self.config.data.image_size,
                self.config.data.image_size
            ).contiguous()
            gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
            all_gather(gathered_samples, img)
            all_samples.extend([img.cpu() for img in gathered_samples])

        samples = torch.concat(
            all_samples,
            dim = 1
        ).reshape(
            self.config.sampling.batch_size * numloop,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size
        )

        if self.local_rank == 0:

            # Save the generated images to a NPY file ................................

            if self.config.sampling.fid:
                samples = (samples * 255).clamp(0, 255).to(torch.uint8)
                samples = samples.permute(0, 2, 3, 1).numpy()[:self.config.sampling.num_images]
                print(f'[INFO] samples shape is {samples.shape}')

                samples_file  = os.path.join(
                    self.config.experiment.root_dir,
                    self.config.experiment.results_dir,
                    self.config.experiment.experiment_name,
                    f'samples_{samples.shape[0]}_epoch_{self.config.sampling.epoch}_w_{self.config.sampling.w}.npz'
                )

                np.savez(samples_file, samples)
            else:
                samples_file  = os.path.join(
                    self.config.experiment.root_dir,
                    self.config.experiment.results_dir,
                    self.config.experiment.experiment_name,
                    f'samples_epoch_{self.config.sampling.epoch}_w_{self.config.sampling.w}.png'
                )
                save_image(
                    samples,
                    samples_file,
                    nrow = self.config.sampling.batch_size // self.config.data.classes,
                )

            # Save each generated image to a PNG file (to calculate metrics)

            print(f'[INFO] Sampled images shape: {samples.shape}')

            for i in range(samples.shape[0]):
                sample_file  = os.path.join(
                    self.config.experiment.root_dir,
                    self.config.experiment.results_dir,
                    self.config.experiment.experiment_name,
                    'generated_images',
                    f'fake_image_epoch_{self.config.sampling.epoch}_{str(i).zfill(5)}.png'
                )

                img = Image.fromarray(samples[i])
                img.save(sample_file)

        # Delete the distributed training processes ............................
        destroy_process_group()

    # ==================================================================
    def restore_checkpoint(self, file_model):
        '''
        Restore a model checkpoint from file.
        '''
        if self.config.experiment.saved_model is not None:
            model_file  = os.path.join(
                self.config.experiment.root_dir,
                self.config.experiment.models_dir,
                self.config.experiment.experiment_name,
                file_model
            )
            if os.path.isfile(model_file) == True:
                loaded_state = torch.load(model_file)

                # load saved epoch and step
                self.epoch = loaded_state['epoch'] + 1
                self.step  = loaded_state['step'] + 1
                # load saved model state
                self.diffusion.model.load_state_dict(loaded_state['unet'])
                self.diffusion.model = self.diffusion.model.to(self.config.device)
                self.cemblayer.load_state_dict(loaded_state['c_embed_layer'])
                self.cemblayer = self.cemblayer.to(self.config.device)

                print(f'[INFO] Loaded checkpoint from {model_file}!')
                return loaded_state
            else:
                self.epoch = 0
                self.step  = 0
                print(f'[ERROR] Checkpoint {model_file} does not exist!')
                return None

    # ==================================================================
    def save_checkpoint(self, model_file):
        '''
        Save a model checkpoint to file.
        '''
        saved_state = {
            'step':          self.step,
            'epoch':         self.epoch,
            'unet':          self.diffusion.model.module.state_dict(),
            'c_embed_layer': self.cemblayer.module.state_dict(),
            'optimizer':     self.optimizer.state_dict(),
            'scheduler':     self.warmUpScheduler.state_dict()
        }
        torch.save(saved_state, model_file)
        print(f'[INFO] Saved checkpoint {model_file}!')

    # ==================================================================
    def set_file_name(self, middle, postfix=None, extension='pth'):
        '''
        Compose a file path given current configuration, epoch, step, postfix and extension.
        '''
        fname_base = f'{self.config.experiment.experiment_name}_epoch{str(self.epoch+1).zfill(3)}_step{str(self.step+1).zfill(8)}'
        if postfix is not None:
            fname_last = f'{fname_base}_{postfix}.{extension}'
        else:
            fname_last = f'{fname_base}.{extension}'

        fname = os.path.join(
            self.config.experiment.root_dir,
            middle,
            self.config.experiment.experiment_name,
            fname_last
        )
        return fname


    # ==================================================================
    def print_model_summary(self):
        '''
        Print a summary of the architecture of the NCSN++ model.
        '''
        batch_size = 8

        # Input shapes:
        #_x:       [BS, CHANNELS, IMG_SIZE, IMG_SIZE]
        # t:       [BS]
        # cond:    [BS, CLASSES]
        unet_aux = Unet(
            in_ch          = self.config.data.channels,
            hidden_ch      = self.config.model.hidden_ch,
            out_ch         = self.config.model.out_ch,
            ch_mult        = self.config.model.ch_mult,
            num_res_blocks = self.config.model.res_blocks,
            cdim           = self.config.model.conditional_dim,
            use_conv       = self.config.model.use_down_conv,
            dropout_prob   = self.config.model.dropout_prob,
            dtype          = self.config.model.dtype,
        )

        # input shape: [batch_size]
        # input values: 0..CLASSES
        cemb_aux = ConditionalEmbedding(
            self.config.data.classes,
            self.config.model.conditional_dim,
            self.config.model.conditional_dim,
        ).to(self.config.device)

        aux_c = torch.randint(low=0, high=self.config.data.classes, size=(batch_size,)).to(self.config.device)

        aux_x = torch.randn(
            (
            batch_size,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size
            )).to(self.config.device)

        aux_t    = torch.randn(batch_size).to(self.config.device)
        aux_cond = torch.randn(batch_size,self.config.data.classes).to(self.config.device)

        sumCemb = summary(
            cemb_aux,
            input_data   = [aux_c],
            col_width    = 16,
            col_names    = ["kernel_size", "output_size", "num_params"],
            row_settings = ["var_names"],
        )
        print(sumCemb)

        sumUNet = summary(
            unet_aux,
            input_data   = [aux_x, aux_t, aux_cond],
            col_width    = 16,
            col_names    = ["kernel_size", "output_size", "num_params"],
            row_settings = ["var_names"],
        )
        print(sumUNet)

        # Delete created models and tensors
        del cemb_aux
        del unet_aux
        del aux_c
        del aux_x
        del aux_t
        del aux_cond


    # ==================================================================
    def save_model_onnx(self):
        '''
        Export the U-Net model to ONNX format.
        '''
        batch_size = 8

        # Input shapes:
        #_x:       [BS, CHANNELS, IMG_SIZE, IMG_SIZE]
        # t:       [BS]
        # cond:    [BS, CLASSES]
        unet_aux = Unet(
            in_ch          = self.config.data.channels,
            hidden_ch      = self.config.model.hidden_ch,
            out_ch         = self.config.model.out_ch,
            ch_mult        = self.config.model.ch_mult,
            num_res_blocks = self.config.model.res_blocks,
            cdim           = self.config.model.conditional_dim,
            use_conv       = self.config.model.use_down_conv,
            dropout_prob   = self.config.model.dropout_prob,
            dtype          = self.config.model.dtype,
        )

        # input shape: [batch_size]
        # input values: 0..CLASSES
        cemb_aux = ConditionalEmbedding(
            self.config.data.classes,
            self.config.model.conditional_dim,
            self.config.model.conditional_dim,
        ).to(self.config.device)

        aux_c = torch.randint(low=0, high=self.config.data.classes, size=(batch_size,)).to(self.config.device)

        aux_x = torch.randn(
            (
            batch_size,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size
            )).to(self.config.device)

        aux_t    = torch.randn(batch_size).to(self.config.device)
        aux_cond = torch.randn(batch_size,self.config.data.classes).to(self.config.device)

        aux_data     = (aux_x, aux_t, aux_cond)  
        #input_names  = [ "X_0", "t" , "cond"] 
        #output_names = [ "Out" ]

        file_onnx  = f'{self.config.experiment.experiment_name}_UNet_architecture.onnx'
        model_file = os.path.join(
            self.config.experiment.root_dir,
            self.config.experiment.models_dir,
            self.config.experiment.experiment_name,
            file_onnx
        )

        print(f"[INFO] creating a ONNX file with the U-Net architecture ... \n{model_file}")
        torch.onnx.dynamo_export(
            unet_aux,
            *aux_data,
        ).save(model_file)

        #input_names  = ["classes"] 
        #output_names = ["class_embeds"]

        file_onnx  = f'{self.config.experiment.experiment_name}_ClassEmbedding_architecture.onnx'
        model_file = os.path.join(
            self.config.experiment.root_dir,
            self.config.experiment.models_dir,
            self.config.experiment.experiment_name,
            file_onnx
        )

        print(f"[INFO] creating a ONNX file with the ClassEmbedding architecture ... \n{model_file}")
        torch.onnx.dynamo_export(
            cemb_aux,
            aux_c,
        ).save(model_file)

        # Delete created models and tensors
        del cemb_aux
        del unet_aux
        del aux_c
        del aux_x
        del aux_t
        del aux_cond


# ==============================================================================
def main():

    config  = setup_environment()

    local_rank = int(os.environ["LOCAL_RANK"])
    if (local_rank == 0):
        print_random_image(config.data.data_path)

    trainer = Trainer(config)

    if config.experiment.mode == "train":

        if (local_rank == 0):
            # Login into Weights & Bias

            wandb.login()

            # Track metadata and hyperparameters with Weights & Bias
            # 
            # Define the experiment: the hyperparameters, the dataset and model name.
            # This information will be stored in a `config` dictionary.

            config_wandb = config

            wandb.init(
                project = 'OUR_WANDB_PROJECT_ID',
                entity  = 'OUR_WANDB_ENTITY',
                config  = config_wandb,
                #id      = config.experiment.experiment_name, # TO CONTINUE LOGGING AFTER TRAINING WAS STOPPED 
                #resume  = 'allow',                           # TO CONTINUE LOGGING AFTER TRAINING WAS STOPPED
            )

        # Train the model
        trainer.train()

        if (local_rank == 0):
            # Mark the Weights and Bias run as finished
            wandb.finish()

    elif config.experiment.mode == "sampling":
        # Sample from the model
        trainer.sample()

    elif config.experiment.mode == "summary":
        # Print a summary of the model architecture
        trainer.print_model_summary()
        # Save the model architecture to ONNX format
        trainer.save_model_onnx()

    else:
        raise ValueError(f"[ERROR] mode {config.mode} is not recognized!")

# ==============================================================================
# EXAMPLE OF COMMAND TO RUN WHEN USING 2 GPUs ON SAME MACHINE:
#   torchrun --standalone --nproc_per_node=2 CFreeGuidance_01_exp02_sampling.py
# Use '--standalone' only when training in a single machine.
# ==============================================================================

if __name__ == "__main__":

    # With 'torchrun' it is not necessary to explicitly launch the group of processes using 'spawn'
    main()
