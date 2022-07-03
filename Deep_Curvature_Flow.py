"""
Author: Chun Lee
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn Navier-Stokes equation (2D).
TRV:
"""
import sys

sys.path.insert(0, '../../Utils')

import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import scipy.io


# from plotting import newfig, savefig

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True)


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))


def figsize(scale, nplots=1):
    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = nplots * fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt


# I make my own newfig and savefig functions
def newfig(width, nplots=1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop=True):
    if crop == True:
        #        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        #        plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))


torch.manual_seed(123456)
np.random.seed(123456)

lamb = 10.0
epsilon = 0.5
gamma_1 = 100.0
gamma_2 = 100.0
gamma_3 = 100.0
#alph = 0.05
beta = 0.5

mu = 10#[0.01,0.5, 5,10]
sigma = 2.5#0.01[0.01, 0.5 , 10]
lamb = 10#[0.05, 0.09, 0.18, 0.25, 0.5 , 2.5, 10]
alph = 10#0.05
tau = 10#0.01[0.01, 0.3, 2.5,10]
gamma = 10
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(3, 50))
        # self.net.add_module('tanh_layer_1', nn.Tanh())
        self.net.add_module('lrelu_layer_1', nn.LeakyReLU())
        for num in range(2,14):
            self.net.add_module('linear_layer_%d' %(num), nn.Linear(50, 50))
            # self.net.add_module('tanh_layer_%d' %(num), nn.Tanh())
            self.net.add_module('lrelu_layer_%d' %(num), nn.LeakyReLU())
        self.net.add_module('linear_layer_50', nn.Linear(50, 8))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        y = self.net(x)
        u, w, b = y[:, 0], y[:, 1], y[:, 2]
        u_g = gradients(u, x)[0]
        w_g = gradients(w, x)[0]
        b_g = gradients(b, x)[0]

        u_t, u_x, u_y = u_g[:, 0], u_g[:, 1], u_g[:, 2]
        w_t, w_x, w_y = w_g[:, 0], w_g[:, 1], w_g[:, 2]
        b_t, b_x, b_y = b_g[:, 0], b_g[:, 1], b_g[:, 2]

        u_xx, u_xy, u_yy = gradients(u_x, x)[0][:, 1], gradients(u_x, x)[0][:, 2], gradients(u_y, x)[0][:, 2]
        w_xx, w_xy, w_yy = gradients(w_x, x)[0][:, 1], gradients(w_x, x)[0][:, 2], gradients(w_y, x)[0][:, 2]

        kappa = (u_xx * (1 + u_y ** 2) + u_yy * (1 + u_x ** 2) - 2 * u_x * u_y * u_xy) / (1 + u_x ** 2 + u_y ** 2) ** (
               3 / 2)
        #kappa = (u_xx * u_yy - u_xy ** 2) / (1 + u_x ** 2 + u_y ** 2) ** 2 #Gaussian curvature
        kappa_g = gradients(kappa, x)[0]
        kappa_t, kappa_x, kappa_y = kappa_g[:, 0], kappa_g[:, 1], kappa_g[:, 2]
        kappa_xx, kappa_xy, kappa_yy = gradients(kappa_x, x)[0][:, 1], gradients(kappa_x, x)[0][:, 2], \
                                       gradients(kappa_y, x)[0][:, 2]
        Laplace_kappa = (kappa_xx * (1 + kappa_y ** 2) + kappa_yy * (
                    1 + kappa_x ** 2) - 2 * kappa_x * kappa_y * kappa_xy) / (1 + kappa_x ** 2 + kappa_y ** 2) ** 2

        g_kappa = 1 + beta * torch.absolute(Laplace_kappa **2)  ##EE
        #g_kappa = torch.log(torch.cosh(Laplace_kappa)) #Geman
        #g_kappa = 1 + alph * torch.absolute(Laplace_kappa) ##TAC
        #g_kappa = 1 + alph * torch.absolute(Laplace_kappa**2)  ##TSC
        #g_kappa = torch.sqrt(1 + 10 * torch.sqrt(Laplace_kappa ** 2))##TRV
        #g_kappa = torch.log(gamma + Laplace_kappa**2) ###log

        Q1 = mu*w + b
        Q1_g = gradients(Q1, x)[0]
        Q1_t, Q1_x, Q1_y = Q1_g[:, 0], Q1_g[:, 1], Q1_g[:, 2]
        q = Q1_x + Q1_y

        loss_1 = w_t + (g_kappa*w*(1+w**2)**(-0.5)+mu*(2*w - (u_x + u_y)) + b + sigma*w_t)/(
            g_kappa*((g_kappa*w*(1+w**2)**(-3/2)) + mu + sigma))
        loss_2 = (lamb + tau)*u + mu*(u_xx + u_yy) - tau*u + q
        loss_3 = b_t - mu*(2*w - (u_x + u_y))
        loss = (loss_1 ** 2).mean() + (loss_2 ** 2).mean() + (loss_3 ** 2).mean()
        return loss

    def loss_bc(self, x_l, x_r, x_up, x_dw):
        y_l, y_r, y_up, y_dw = self.net(x_l), self.net(x_r), self.net(x_up), self.net(x_dw)
        # y_l, y_r, y_up, y_dw = self.conv_last(x_l), self.conv_last(x_r), self.conv_last(x_up), self.conv_last(x_dw)
        #u, w, b, c, d
        u_l, w_l, b_l = y_l[:, 0], y_l[:, 1], y_l[:, 2]
        u_r, w_r, b_r= y_r[:, 0], y_r[:, 1], y_r[:, 2]
        u_up, w_up, b_up= y_up[:, 0], y_up[:, 1], y_up[:, 2]
        u_dw, w_dw, b_dw = y_dw[:, 0], y_dw[:, 1], y_dw[:, 2]

        loss = ((u_l - u_r) ** 2).mean() + ((w_l - w_r) ** 2).mean() + ((w_l - w_r) ** 2).mean() + \
               ((b_l - b_r) ** 2).mean() +\
               ((u_up) ** 2).mean() + ((u_dw) ** 2).mean() + ((w_up) ** 2).mean() + ((w_dw) ** 2).mean() + \
               ((b_up) ** 2).mean() + ((b_dw) ** 2).mean()

        return loss

    def loss_ic(self, x_i, u_i, w_i, b_i):
        y_pred = self.net(x_i)
        # y_pred = self.conv_last(x_i)
        u_i_pred, w_i_pred, b_i_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        return ((u_i_pred - u_i) ** 2).mean() + ((w_i_pred - w_i) ** 2).mean() + \
               ((b_i_pred - b_i) ** 2).mean()
nu=0.01
def AC_2D_init(x):
    ## bubble
    u_init = 0.45 * (np.exp(-15 * ((x[:, 1] - 0.3) ** 2 + (x[:, 2] - 0.3) ** 2)) + \
                     np.exp(-25 * ((x[:, 1] - 0.5) ** 2 + (x[:, 2] - 0.75) ** 2)) + \
                     np.exp(-30 * ((x[:, 1] - 0.8) ** 2 + (x[:, 2] - 0.35) ** 2))
                     )
    w_init = np.zeros((x.shape[0]))
    b_init = np.zeros((x.shape[0]))
    noise = 0.9# [0.25, 0.5, 0.75, 0.9]
    u_init = u_init + noise * np.std(u_init) * np.random.randn(u_init.shape[0])
    return u_init, w_init, b_init

loss_history = {
    "loss_pde": [],
    "loss_ic": [],
    "loss_bc": [],
    "train_loss": []
}
def main():
    ##parameters
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    print(device)
    epochs = 300
    lr = 0.001

    num_x = 100
    num_y = 100
    num_t = 100
    num_b_train = 50    # boundary sampling points
    num_f_train = 50000   # inner sampling points
    num_i_train = 5000   # initial sampling points
    num_t_train = 40


    x = np.linspace(-1, 1, num=num_x)
    y = np.linspace(-1, 1, num=num_y)
    t = np.linspace(0, 5, num=num_t)[:, None]
    x_grid, y_grid = np.meshgrid(x, y)
    x_2d = np.concatenate((x_grid.flatten()[:, None], y_grid.flatten()[:, None]), axis=1)
    ## initialization
    xt_init = np.concatenate((np.zeros((num_x*num_y, 1)), x_2d), axis=1)
    u_init, w_init, b_init = AC_2D_init(xt_init)

    ## save init fig
    fig, ax = newfig(2.0, 1.1)
    ax.axis('off')
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(u_init.reshape(num_x, num_y), interpolation='nearest', cmap='Spectral',
                  # extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    fig.colorbar(h)
    ax.plot(xt_init[:, 1], xt_init[:, 2], 'kx', label='Data (%d points)' % (xt_init.shape[0]), markersize=4,
            clip_on=False)
    line = np.linspace(xt_init.min(), xt_init.max(), 2)[:, None]
    fig.savefig('Figures-1/u_init.png', dpi=300)
    #fig.savefig('Figures-1/u_init.eps', dpi=300)
    fig.savefig('Figures-1/u_init.pdf', dpi=300)

    x_2d_ext = np.tile(x_2d, [num_t, 1])
    t_ext = np.zeros((num_t * num_x * num_y, 1))
    for i in range(0, num_t):
        t_ext[i * num_x * num_y:(i + 1) * num_x * num_y, :] = t[i]
        xt_2d_ext = np.concatenate((t_ext, x_2d_ext), axis=1)

    ## sampling
    id_f = np.random.choice(num_x * num_y * num_t, num_f_train, replace=False)
    id_b = np.random.choice(num_x, num_b_train, replace=False)  ## Dirichlet
    id_i = np.random.choice(num_x * num_y, num_i_train, replace=False)
    id_t = np.random.choice(num_t, num_t_train, replace=False)

    ## boundary
    t_b = t[id_t, :]
    t_b_ext = np.zeros((num_t_train * num_b_train, 1))
    for i in range(0, num_t_train):
        t_b_ext[i * num_b_train:(i + 1) * num_b_train, :] = t_b[i]
    x_up = np.vstack((x_grid[-1, :], y_grid[-1, :])).T
    x_dw = np.vstack((x_grid[0, :], y_grid[0, :])).T
    x_l = np.vstack((x_grid[:, 0], y_grid[:, 0])).T
    x_r = np.vstack((x_grid[:, -1], y_grid[:, -1])).T

    x_up_sample = x_up[id_b, :]
    x_dw_sample = x_dw[id_b, :]
    x_l_sample = x_l[id_b, :]
    x_r_sample = x_r[id_b, :]

    x_up_ext = np.tile(x_up_sample, (num_t_train, 1))
    x_dw_ext = np.tile(x_dw_sample, (num_t_train, 1))
    x_l_ext = np.tile(x_l_sample, (num_t_train, 1))
    x_r_ext = np.tile(x_r_sample, (num_t_train, 1))

    xt_up = np.hstack((t_b_ext, x_up_ext))
    xt_dw = np.hstack((t_b_ext, x_dw_ext))
    xt_l = np.hstack((t_b_ext, x_l_ext))
    xt_r = np.hstack((t_b_ext, x_r_ext))

    xt_i = xt_init[id_i, :]
    xt_f = xt_2d_ext[id_f, :]

    ## set data as tensor and send to device
    xt_f_train = torch.tensor(xt_f, requires_grad=True, dtype=torch.float32).to(device)
    # x_test = torch.tensor(xt_2d_ext, requires_grad=True, dtype=torch.float32).to(device)
    xt_i_train = torch.tensor(xt_init, requires_grad=True, dtype=torch.float32).to(device)
    x_i_train = torch.tensor(x_2d, requires_grad=True, dtype=torch.float32).to(device)

    u_i_train = torch.tensor(u_init, dtype=torch.float32).to(device)
    w_i_train = torch.tensor(w_init, dtype=torch.float32).to(device)
    b_i_train = torch.tensor(b_init, dtype=torch.float32).to(device)

    xt_l_train = torch.tensor(xt_l, requires_grad=True, dtype=torch.float32).to(device)
    xt_r_train = torch.tensor(xt_r, requires_grad=True, dtype=torch.float32).to(device)
    xt_up_train = torch.tensor(xt_up, requires_grad=True, dtype=torch.float32).to(device)
    xt_dw_train = torch.tensor(xt_dw, requires_grad=True, dtype=torch.float32).to(device)

    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    import torch_optimizer as optim
    optimizer = optim.NovoGrad(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        grad_averaging=False,
        amsgrad=False,
    )
    #optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    train_loss = np.zeros((epochs, 1))
    # training
    def train(epoch):
        model.train()

        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(xt_f_train)
            loss_bc = model.loss_bc(xt_l_train, xt_r_train, xt_up_train, xt_dw_train)
            loss_ic = model.loss_ic(xt_i_train, u_i_train, w_i_train, b_i_train)
            loss = loss_pde + loss_bc + 100 * loss_ic
            print(f'epoch {epoch} loss_pde:{loss_pde:6f}, loss_bc:{loss_bc:6f}, loss_ic:{loss_ic:6f}')
            loss_history["loss_pde"].append(loss_pde.item())
            loss_history["loss_bc"].append(loss_bc.item())
            loss_history["loss_ic"].append(loss_ic.item())
            train_loss[epoch, 0] = loss
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        print(f'epoch {epoch}: loss {loss_value:.6f}')
        loss_history["train_loss"].append(loss_value)

    print('start training...')
    tic = time.time()
    #for epoch in range(1, epochs + 1):
    for epoch in range(0, epochs):
        train(epoch)
        print(f'Train Epoch: {epoch + 1}, Train Loss: {train_loss[epoch, 0]:6f}', flush=True)
    toc = time.time()
    print(f'total training time: {toc - tic}')
    np.savetxt("Figures-1/01_loss_1.txt", train_loss)

    ##plot loss progress
    pparam = dict(xlabel='Epochs', ylabel='MSELoss')
    with plt.style.context(['high-vis', 'grid']):  # 'science', 'high-vis'
        # fig, ax = newfig(2.0, 1.1)
        fig, ax = plt.subplots()
        # ax.axis('off')
        # plt.title("loss")
        plt.plot(range(1, epochs + 1), loss_history["train_loss"], label="total loss", linewidth=3)
        plt.plot(range(1, epochs + 1), loss_history["loss_pde"], label="loss pde", linewidth=3)
        plt.plot(range(1, epochs + 1), loss_history["loss_ic"], label="loss ic", linewidth=3)
        plt.plot(range(1, epochs + 1), loss_history["loss_bc"], label="loss bc", linewidth=3)
        ax.legend()
        ax.autoscale(tight=True)
        ax.set(**pparam)
        # fig.savefig('Figures-1/Loss.png', dpi=300)
        fig.savefig('Figures-1/Loss.pdf')
        plt.show()

    ## test
    u_test = np.zeros((num_t, num_x, num_y))
    for i in range(0, num_t):
        xt = np.concatenate((t[i] * np.ones((num_x * num_y, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.float32).to(device)
        y_pred = model(xt_tensor)
        u_test[i, :, :] = to_numpy(y_pred[:, 0]).reshape(num_x, num_y)

        fig, ax = newfig(2.0, 1.1)
        ax.axis('off')
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(u_test[i, :, :], interpolation='nearest', cmap='Spectral',
                      # extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto')
        fig.colorbar(h)
        ax.plot(xt[:, 1], xt[:, 2], 'kx', label='Data (%d points)' % (xt.shape[0]), markersize=4, clip_on=False)
        line = np.linspace(xt.min(), xt.max(), 2)[:, None]
        fig.savefig('Figures-1/u_' + str(i + 1000) + '.png', dpi=300)
        fig.savefig('Figures-1/u_' + str(i + 1000) + '.pdf', dpi=300)
        #fig.savefig('Figures-1/u_' + str(i + 1000) + '.eps', dpi=300)

if __name__ == '__main__':
    main()

