import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


device = 'cuda' if torch.cuda.is_available() else 'cpu'

DIMS = 100   # number of dimensions that xn has
WSUM = 5    # number of waves added together to make a splotch
EPSILON = 0.0025 # rate at which xn controlls splotch strength
TRAIN_TIME = 10_00 # number of iterations to train for
LEARN_RATE = 0.2   # learning rate

torch.random.manual_seed(1729)
np.random.seed(1729)

# knlist and k0list are integers, so the splotch functions are periodic
knlist = torch.randint(-2, 3, (DIMS, WSUM, DIMS)) # wavenumbers : list (controlling dim, wave id, k component)
k0list = torch.randint(-2, 3, (DIMS, WSUM))       # the x0 component of wavenumber : list (controlling dim, wave id)
slist = torch.randn((DIMS, WSUM))                # sin coefficients for a particular wave : list(controlling dim, wave id)
clist = torch.randn((DIMS, WSUM))                # cos coefficients for a particular wave : list (controlling dim, wave id)

# initialize x0, xn
x0 = torch.zeros(1, requires_grad=True)
xn = torch.zeros(DIMS, requires_grad=True)

# numpy arrays for plotting:
x0_hist = np.zeros((TRAIN_TIME,))
xn_hist = np.zeros((TRAIN_TIME, DIMS))

for tensor in [knlist, k0list, slist, clist, x0, xn]:
    tensor.to(device)

t = 0 # fix pyright unbound variable
try:
# train:
    for t in trange(TRAIN_TIME):
        ### model: 
        wavesum = torch.sum(knlist*xn, dim=2) + k0list*x0
        splotch_n = torch.sum(
                (slist*torch.sin(wavesum)) + (clist*torch.cos(wavesum)),
                dim=1)
        foreground_loss = EPSILON * torch.sum(xn * splotch_n)
        loss = foreground_loss - x0
        ###
        if t % 100 == 0:
            tqdm.write(f'{t}: loss {loss.item():.3f} x0 {x0.item():.3f} (max, min) xn ({xn.max():.3f}, {xn.min():.3f})')
        loss.backward()
        with torch.no_grad():
            # constant step size gradient descent, with some noise thrown in
            vlen = torch.sqrt(x0.grad*x0.grad + torch.sum(xn.grad*xn.grad))
            x0 -= LEARN_RATE*(x0.grad/vlen + torch.randn(1)/np.sqrt(1.+DIMS))
            xn -= LEARN_RATE*(xn.grad/vlen + torch.randn(DIMS)/np.sqrt(1.+DIMS))
        x0.grad.zero_()
        xn.grad.zero_()
        x0_hist[t] = x0.detach().numpy()
        xn_hist[t] = xn.detach().numpy()
except KeyboardInterrupt:
    print(f'\nABORTED AFTER {t}/{TRAIN_TIME} ITERATIONS ({100*t/TRAIN_TIME:.3f}%)')

print('Saving histories...')
prefix = f'dims{DIMS}wsum{WSUM}eps{EPSILON}time{TRAIN_TIME}lr{LEARN_RATE}'
np.savez_compressed(f'{prefix}.npz', x0_hist=x0_hist, xn_hist=xn_hist)

print('Drawing plots...')

plt.plot(x0_hist)
plt.xlabel('number of steps')
plt.ylabel('x0')
plt.savefig(f'{prefix}-plot1.png')
plt.show()


idx = np.arange(xn_hist.shape[1])
idx_perm = np.random.choice(idx, size=min(DIMS, 16), replace=False)

for i in idx_perm.tolist() + [xn_hist.argmin(axis=1), xn_hist.argmax(axis=1)]:
    plt.plot(xn_hist[:,i])

plt.xlabel('number of training steps')
plt.ylabel('xn')
plt.savefig(f'{prefix}-plot2.png')
plt.show()

