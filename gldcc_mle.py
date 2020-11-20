'''
python3 gldcc_mle.py --datasource=omniglot --K=8 --L=4 --delta=0.5 --num-samples-per-class=16 --min-num-cls=5 --max-num-cls=10 --alpha0=0.3 --tau=1e6 --num-episodes-per-epoch=10000 --minibatch=20 --num-epochs=10 --resume-epoch=0

python3 gldcc_mle.py --datasource=miniImageNet_640 --K=8 --L=4 --delta=0.5 --num-samples-per-class=16 --min-num-cls=5 --max-num-cls=10 --alpha0=0.2 --tau=1e6 --num-episodes-per-epoch=10000 --minibatch=20 --num-epochs=10 --resume-epoch=0
'''
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import random

import pickle
import typing

import argparse

from EpisodeGenerator import ImageFolderGenerator, ImageEmbeddingGenerator
from utils import expected_log_dirichlet

np.set_printoptions(precision=3)

# -------------------------------------------------------------------------------------------------
# Setup input parser
# -------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables')

parser.add_argument('--datasource', type=str, default='miniImageNet_ae', help='Datasource: omniglot, miniImageNet, miniImageNet_640')
parser.add_argument('--ds-root-folder', type=str, default='../datasets/', help='Root folder containing a folder of the data set')
parser.add_argument('--logdir-root', type=str, default='/media/n10/Data/Topic_models', help='Directory to store logs and models')

parser.add_argument('--L', type=int, default=3, help='Number of task-themes')
parser.add_argument('--K', type=int, default=8, help='Number of image-themes')

parser.add_argument('--num-samples-per-class', type=int, default=16, help='Number of training examples per class')
parser.add_argument('--min-num-cls', type=int, default=5, help='Minimum number of classes per episode')
parser.add_argument('--max-num-cls', type=int, default=10, help='Maximum number of classes per episode')

parser.add_argument('--minibatch', type=int, default=2, help='Mini-batch size or the number of tasks used for an update')
parser.add_argument('--alpha0', type=float, default=0.2, help='Initial value for alpha')
parser.add_argument('--tau', type=float, default=1e6, help='Parameter to calculate learning rate')

parser.add_argument('--resume-epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

parser.add_argument('--num-episodes-per-epoch', type=int, default=1000, help='Number of tasks in each epoch')
parser.add_argument('--num-epochs', type=int, default=1000, help='How many \'epochs\' are going to be?')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--subset',type=str, default='test', help='Subset used for testing')

parser.add_argument('--delta', type=float, default=0.1, help='Dirichlet parameter of document-topic proportion')

args = parser.parse_args()
config = {}
for key in args.__dict__:
    config[key] = args.__dict__[key]
# -------------------------------------------------------------------------------------------------
# Setup CPU or GPU
# -------------------------------------------------------------------------------------------------
gpu_id = 0
config['device'] = torch.device('cuda:{0:d}'.format(gpu_id) \
    if torch.cuda.is_available() else torch.device('cpu'))

# -------------------------------------------------------------------------------------------------
# Parse dataset and related variables
# -------------------------------------------------------------------------------------------------
print('Data source = {0:s}'.format(config['datasource']))
if config['datasource'] == 'omniglot':
    img_size = (1, 28, 28)
    config['D'] = np.prod(img_size)
    eps_generator = ImageFolderGenerator(
        root=os.path.join(config['ds_root_folder'], config['datasource']),
        train_subset=True,
        suffix='.png',
        min_num_cls=config['min_num_cls'],
        max_num_cls=config['max_num_cls'],
        k_shot=config['num_samples_per_class'],
        expand_dim=False,
        load_images=True
    )
elif config['datasource'] in ['miniImageNet_640', 'tierImageNet_640']:
    config['D'] = 640
    eps_generator = ImageEmbeddingGenerator(
        root=os.path.join(config['ds_root_folder'], config['datasource']),
        train_subset=True,
        suffix='.pkl',
        min_num_cls=config['min_num_cls'],
        max_num_cls=config['max_num_cls'],
        k_shot=config['num_samples_per_class'],
        expand_dim=False,
        load_images=True
    )
else:
    raise NotImplementedError

# -------------------------------------------------------------------------------------------------
# Parse training parameters
# -------------------------------------------------------------------------------------------------
print('Mini-batch size = {0:d}'.format(config['minibatch']))
print('Number of task-themes = {0:d}\nNumber of image-themes = {1:d}'.format(config['L'], config['K']))

# Tolerance
tols = {
    'gamma': 1e-5,
    'eta': 1e-4,
    'lambda': 1e-5
}
num_steps = 100

cov_reg = 1e-6 * torch.eye(n=config['D'], device=config['device'])

# -------------------------------------------------------------------------------------------------
# Setup destination folder
# -------------------------------------------------------------------------------------------------
config['logdir'] = '{0:s}/gldcc_mle_{1:s}_L{2:d}_K{3:d}'.format(
    config['logdir_root'], config['datasource'], config['L'], config['K'])
print('Log file at {0:s}'.format(config['logdir']))
if not os.path.exists(config['logdir']):
    from pathlib import Path
    Path(config['logdir']).mkdir(parents=True, exist_ok=True)
    print('No folder found. Creating new folder for storage')
    print(config['logdir'])

# -------------------------------------------------------------------------------------------------
# Initialize/Load paramters for training
# -------------------------------------------------------------------------------------------------
def initialize_parameters(config: dict) -> typing.Tuple[torch.Tensor, torch.distributions.MultivariateNormal]:
    """Initialize alpha and the parameters of normal-Wishart distribution

    Args:
        config: a dictionary containing the configuration

    Returns:
        alpha: an L-by-K matrix
        gmm: K D-variate Gaussian components
    """
    print('Initialize parameters...')
    # INITIALIZATION - document-invariant parameters
    alpha = config['alpha0'] + 0.1 * torch.rand(size=(config['L'], config['K']), dtype=torch.float, device=config['device'])

    # initialize Gaussian components
    gmm = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=0.05 * torch.randn(size=(config['K'], config['D']), device=config['device']),
        scale_tril=10 * torch.diag_embed(input=torch.ones(size=(config['K'], config['D']), device=config['device']))
    )

    return alpha, gmm

def load_parameters(config: dict) -> typing.Tuple[torch.Tensor, torch.distributions.MultivariateNormal]:
    """Load alpha and the parameters of a normal-Wishart distribution

    Args:
        config: a dictionary containing the configuration

    Returns:
        alpha: an L-by-K matrix
        gmm: K D-variate Gaussian components
    """
    print('Resume data from epoch = {0:d}'.format(config['resume_epoch']))
    checkpoint_filename = 'Epoch_{0:d}.pt'.format(config['resume_epoch'])
    checkpoint_fullpath = os.path.join(config['logdir'], checkpoint_filename)
    print('Load data from {0:s}'.format(checkpoint_fullpath))
    if torch.cuda.is_available():
        saved_checkpoint = torch.load(
            checkpoint_fullpath,
            map_location=lambda storage,
            loc: storage.cuda(gpu_id)
        )
    else:
        saved_checkpoint = torch.load(
            checkpoint_fullpath,
            map_location=lambda storage,
            loc: storage
        )
    alpha = saved_checkpoint['alpha']
    gmm = saved_checkpoint['gmm']

    return alpha, gmm

print()

def training() -> None:
    minibatch_print = np.lcm(config['minibatch'], 200)

    # initialize/load parameters
    if config['resume_epoch'] == 0:
        get_parameters_ = initialize_parameters
    else:
        get_parameters_ = load_parameters

    alpha, gmm = get_parameters_(config=config)

    print()
    try:
        # tensorboard to monitor
        tb_writer = SummaryWriter(
            log_dir=config['logdir'],
            purge_step=config['resume_epoch'] * config['num_episodes_per_epoch'] // minibatch_print \
                if config['resume_epoch'] > 0 else None
        )

        # task_count = -1 if config['resume_epoch'] == 0 else 0
        for epoch in range(config['resume_epoch'], config['resume_epoch'] + config['num_epochs']):
            task_count = 0
            L_monitor = 0

            # initialize variables that accumulate for new update
            gmm_accum = {'mean': 0, 'cov': 0}

            # initialize variable to store Newton update for alpha
            Hinv_g_accum = 0

            while (task_count < config['num_episodes_per_epoch']):
                x_t = eps_generator.generate_episode()
                x = torch.tensor(x_t, dtype=torch.float, device=config['device']) # (C, N, D)
                if x.ndim > 3:
                    x = torch.flatten(input=x, start_dim=2, end_dim=-1)

                variational_parameters = E_step(x=x, alpha=alpha, gmm=gmm, config=config)
                gmm_task, singular_flag = M_step(x=x, variational_parameters=variational_parameters, config=config)

                if singular_flag:
                    print('{0:d}, {1:d}, Singularity detected!'.format(epoch, task_count))
                    continue

                # accumulate for online update
                gmm_accum['mean'] = gmm_accum['mean'] + gmm_task.loc
                gmm_accum['cov'] = gmm_accum['cov'] + gmm_task.covariance_matrix #- cov_reg
                
                # calculation for alpha
                Hinv_g = get_Hinv_g(
                    gamma=torch.exp(input=variational_parameters['log_gamma']).float(),
                    eta=torch.exp(input=variational_parameters['log_eta']).float(),
                    alpha=alpha
                )
                Hinv_g_accum += Hinv_g

                L = get_elbo(x=x, alpha=alpha, variational_parameters=variational_parameters, gmm=gmm_task)
                L /= (x.shape[0] * x.shape[1] * config['K'])

                if torch.isnan(input=L):
                    raise ValueError('ELBO is NaN')

                L_monitor += L.item()

                # UPDATE
                task_count = task_count + 1
                if (task_count % config['minibatch'] == 0):
                    # average to obtain parameters for a minibatch
                    for key in gmm_accum:
                        gmm_accum[key] = gmm_accum[key] / config['minibatch']
                    Hinv_g_accum /= config['minibatch']

                    # calculate learning rate
                    lr = 1 / np.sqrt(config['tau'] + (epoch * config['num_episodes_per_epoch'] + task_count) / config['minibatch'])

                    # update
                    mean_temp = (1 - lr) * gmm.loc + lr * gmm_accum['mean']
                    cov_temp = (1 - lr) * gmm.covariance_matrix + lr * gmm_accum['cov']
                    gmm = torch.distributions.multivariate_normal.MultivariateNormal(
                        loc=mean_temp,
                        covariance_matrix=cov_temp
                    )

                    Hinv_g_accum = Hinv_g_accum / torch.linalg.norm(input=Hinv_g_accum, dim=-1, keepdim=True)
                    alpha = alpha - lr * Hinv_g_accum
                    
                    # if (alpha <= 0).any(): # or (alpha >= 1.1).any():
                    #     alpha = 0.1 * torch.rand(size=(config['L'], config['K']), device=config['device'])
                    
                    if (task_count % minibatch_print == 0):
                        L_monitor /= minibatch_print
                        global_step = (epoch * config['num_episodes_per_epoch'] + task_count) // minibatch_print

                        tb_writer.add_scalar(tag='Loss', scalar_value=-L_monitor, global_step=global_step)

                        log_Nk = torch.logsumexp(input=variational_parameters['log_r'], dim=(0, 1))
                        Nk = torch.exp(input=log_Nk)
                        Nk = (Nk / torch.sum(input=Nk, dim=-1) * 100).cpu().numpy()
                        for k in range(config['K']):
                            tb_writer.add_scalar(tag='Nk/{0:d}'.format(k), scalar_value=Nk[k], global_step=global_step)

                        alpha_min = torch.min(input=alpha, dim=-1)[0].cpu().numpy()
                        alpha_max = torch.max(input=alpha, dim=-1)[0].cpu().numpy()
                        log_Nl = torch.logsumexp(input=variational_parameters['log_eta'], dim=0)
                        Nl = torch.exp(input=log_Nl)
                        Nl = (Nl / torch.sum(input=Nl, dim=-1) * 100).cpu().numpy()
                        for l in range(config['L']):
                            tb_writer.add_scalar(tag='alpha_min/{0:d}'.format(l), scalar_value=alpha_min[l], global_step=global_step)
                            tb_writer.add_scalar(tag='alpha_max/{0:d}'.format(l), scalar_value=alpha_max[l], global_step=global_step)
                            tb_writer.add_scalar(tag='Nl/{0:d}'.format(l), scalar_value=Nl[l], global_step=global_step)

                        L_monitor = 0
                
                if (task_count >= config['num_episodes_per_epoch']):
                    break
            
            # task_count = 0

            checkpoint = {
                'alpha': alpha,
                'gmm': gmm
            }
            checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch + 1)
            torch.save(checkpoint, os.path.join(config['logdir'], checkpoint_filename))
            del checkpoint
            print('SAVING parameters into {0:s}'.format(checkpoint_filename))
            print('----------------------------------------\n')
    finally:
        # --------------------------------------------------
        # clean up
        # --------------------------------------------------
        tb_writer.close()
        print('\nClose tensorboard summary writer')
    
    return None

def E_step(x: torch.Tensor, alpha: torch.Tensor, gmm: torch.distributions.MultivariateNormal, config: dict) -> dict:
    """E-step to calculate task-related variational parameters

    Args:
        x: input data in shape (C, N, D)
        alpha: concentration matrix in shape (L, K)
        gmm: K D-variate Gaussian components
        config: dictionary containing configuration parameters

    Returns: a dictionary containing task-related variational parameters
    """
    num_classes = x.shape[0]

    # initialize the parameter LAMBDA of document-topic mixture PHI
    lambda_ = torch.ones(config['L'], device=config['device']) * config['delta'] + num_classes / config['L']

    # initialize the parameter ETA of document-topic assignment Y
    eta = torch.ones(size=(num_classes, config['L']), device=config['device']) / config['L']

    # initialize the paramter GAMMA of word-topic mixture THETA
    gamma = torch.ones(size=(num_classes, config['K']), device=config['device']) * \
        alpha[np.random.randint(low=0, high=config['L'], size=num_classes)] \
            + config['num_samples_per_class'] / config['K']
    
    # calculate log-likelihood of x
    log_prob = gmm.log_prob(value=x[:, :, None, :]) # (C, N, K)

    lambda_count = 0
    dlambda = 1
    while (dlambda > tols['lambda']) and (lambda_count < num_steps):
        log_theta_tilde = expected_log_dirichlet(concentration=gamma) #(C, K)

        # un-normalized r
        r_unnormalized = log_prob + log_theta_tilde[:, None, :] #(C, N, K)

        # normalize r
        log_r = torch.nn.functional.log_softmax(input=r_unnormalized, dim=-1)

        # calculate new gamma
        log_Nck = torch.logsumexp(input=log_r, dim=1) # (C, K)
        log_gamma_temp = torch.matmul(input=eta, other=alpha - 1) # (C, K)
        log_gamma_temp = torch.log1p(input=log_gamma_temp) # (C, K)
        log_gamma = torch.logaddexp(input=log_Nck, other=log_gamma_temp) # (C, K)
        gamma = torch.exp(input=log_gamma).float()

        # calculate new eta
        log_phi_tilde = expected_log_dirichlet(concentration=lambda_) # (L, )
        eta_unnormalized = log_phi_tilde - torch.distributions.dirichlet.Dirichlet._log_normalizer(self=None, x=alpha) + torch.matmul(input=log_theta_tilde, other=alpha.T - 1) # (C, L)
        # normalize eta
        log_eta = torch.nn.functional.log_softmax(input=eta_unnormalized, dim=-1) # (C, L)
        eta = torch.exp(input=log_eta).float()
    
        # store previous lambda_
        lambda__ = lambda_ + 0

        # calculate new lambda_
        log_lambda = torch.logaddexp(
            input=torch.tensor(np.log(config['delta']), device=config['device']),
            other=torch.logsumexp(input=log_eta, dim=0)
        )
        lambda_ = torch.exp(input=log_lambda).float()
        
        dlambda = torch.mean(torch.abs(lambda_ - lambda__))
        if torch.isnan(dlambda):
            raise ValueError('dlambda is NaN')

        lambda_count += 1
    
    variational_parameters = {
        'log_r': log_r,
        'log_gamma': log_gamma,
        'log_eta': log_eta,
        'log_lambda': log_lambda
    }

    return variational_parameters

def M_step(x: torch.Tensor, variational_parameters: dict, config: dict) -> typing.Tuple[torch.distributions.MultivariateNormal, bool]:
    """Calculate the variational parameters of normal-Wishart distribution

    Args:
        x: input data (C, N, D)
        variational_parameters: task-related variational parameters obtained from E-step
        config:

    Returns: MLE of means and covariance matrices
    """
    singular_flag = False

    # AUXILLIARY STATISTICS
    log_Nk = torch.logsumexp(input=variational_parameters['log_r'], dim=(0, 1)) # (K, )
    Nk = torch.exp(input=log_Nk).float() + 1e-6
    # Nk.data = torch.clamp(input=Nk.data, min=1e-6)
    r = torch.exp(input=variational_parameters['log_r']).float() # (C, N, K)

    x_bar_matrix = torch.matmul(input=r[:, :, :, None], other=x[:, :, None, :]) # (C, N, K, D)
    x_bar = torch.sum(input=x_bar_matrix, dim=(0, 1)) / Nk[:, None] # (K, D)
    dx = x[:, :, None, :, None] - x_bar[:, :, None] # (C, N, K, D, 1)
    Sk = torch.matmul(input=dx, other=torch.transpose(input=dx, dim0=-2, dim1=-1)) # (C, N, K, D, D)
    Sk = torch.sum(input=r[:, :, :, None, None] * Sk, dim=(0, 1)) # (K, D, D)

    try:
        gmm_mle = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=x_bar,
            covariance_matrix=Sk / Nk[:, None, None] + cov_reg
        )
    except RuntimeError:
        singular_flag = True
        gmm_mle = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.randn_like(x_bar),
            scale_tril=torch.diag_embed(input=1 + 0.1 * torch.rand(size=(config['K'], config['D']), dtype=torch.float, device=config['device']))
        )
    finally:
        return gmm_mle, singular_flag

def get_elbo(x: torch.Tensor, alpha: torch.Tensor, variational_parameters: dict, gmm: torch.distributions.MultivariateNormal) -> torch.Tensor:
    """
    """
    # convert log to actual form
    r = torch.exp(input=variational_parameters['log_r']).float()
    gamma = torch.exp(input=variational_parameters['log_gamma']).float()
    eta = torch.exp(input=variational_parameters['log_eta']).float()
    lambda_ = torch.exp(input=variational_parameters['log_lambda']).float()

    delta = torch.tensor([config['delta']] * config['L'], dtype=torch.float, device=config['device'])

    # auxilliary variables
    log_theta_tilde = expected_log_dirichlet(concentration=gamma) # (C, K)
    log_phi_tilde = expected_log_dirichlet(concentration=lambda_)

    # Eq [log p(z | theta)]
    log_pz = torch.sum(input=r * log_theta_tilde[:, None, :])

    # Eq [log p(theta | alpha, y)]
    log_ptheta = torch.sum(input=eta * (- torch.distributions.Dirichlet._log_normalizer(self=None, x=alpha[None, :, :]) \
        + torch.matmul(input=log_theta_tilde, other=alpha.T - 1)))

    # Eq [log p(y | phi)]
    log_py = torch.sum(input=eta * log_phi_tilde[None, :])

    # Eq [log p(phi | delta)]
    log_pphi = - torch.distributions.Dirichlet._log_normalizer(self=None, x=delta) \
        + torch.sum(input=(delta - 1) * log_phi_tilde, dim=-1)

    # Eq [log p(x | z, mean, precision)]
    log_prob = gmm.log_prob(value=x[:, :, None, :]) # (C, N, K)
    log_px = torch.sum(input=r * log_prob)

    # Eq [log q(z)]
    log_qz = torch.sum(input=r * variational_parameters['log_r'])

    # Eq [log q(theta)]
    log_qtheta = torch.sum(input=-torch.distributions.dirichlet.Dirichlet(concentration=gamma).entropy())

    # Eq [log q(y)]
    log_qy = torch.sum(input= eta * variational_parameters['log_eta'])

    # Eq [log q(phi)]
    log_qphi = - torch.distributions.dirichlet.Dirichlet(concentration=lambda_).entropy()

    L = log_pz + log_ptheta + log_py + log_pphi + log_px - log_qz - log_qtheta - log_qy - log_qphi

    return L

def get_Hinv_g(gamma: torch.Tensor, eta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    """
    log_theta_tilde = expected_log_dirichlet(concentration=gamma) #(C, K)

    # calculate first derivative
    g = torch.matmul(input=eta.T, other=log_theta_tilde) # (L, K)
    g = g  - expected_log_dirichlet(concentration=alpha) * torch.sum(input=eta, dim=0)[:, None] # (L, K)

    # Q = diag_embed(L, K) = (L, K, K). Here, Q = (L, K)
    Q = - torch.sum(input=eta, dim=0)[:, None] * torch.polygamma(input=alpha, n=1) # (L, K)

    u = torch.sum(input=eta, dim=0) * torch.polygamma(input=torch.sum(input=alpha, dim=-1), n=1) # (L, )

    b = torch.sum(input=g * Q, dim=-1) / (1 / u + torch.sum(input=1 / Q, dim=-1)) # (L, )

    Hinv_g = (g - b[:, None]) / Q # (L, K)

    return Hinv_g

if __name__ == "__main__":
    if config['train_flag']:
        training()
    else:
        raise NotImplementedError
