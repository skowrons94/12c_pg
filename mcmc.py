import numpy as np

from scipy import stats
from brick.azr import AZR

# We read the .azr file and set the external capture file to speed up the calculation
azr = AZR('12c_pg.azr')

# We'll read the data from the output file since it's already in the center-of-mass frame
data    = np.vstack([np.loadtxt('output/'+f) for f in azr.config.data.output_files])
x       = data[:, 0]
y       = data[:, 5]
dy_bare = data[:, 6]

# Prior disributions for each sampled parameter (in AZURE2 order)
priors = [
    stats.norm(1.63,0.12),

    stats.uniform(2.30, 0.10), stats.uniform(0, 100000), stats.uniform(-10, 20),
    stats.uniform(-1000000,2000000),

    stats.uniform(3.45, 0.10), stats.uniform(0, 100000), stats.uniform(-10, 20), stats.uniform(-10, 20),
    stats.uniform(-1000000,2000000), stats.uniform(-1000000,2000000),

    stats.uniform(3.50, 0.10), stats.uniform(0, 100000),
    
    stats.norm(1.0,0.05),
    stats.norm(1.0,0.05),
    stats.norm(1.0,0.05),
    stats.norm(1.0,0.069),
    stats.norm(1.0,0.079),
    stats.norm(1.0,0.10),
    stats.norm(1.0,0.10),
    stats.norm(1.0,0.10),
    stats.norm(1.0,0.10),
    stats.norm(1.0,0.10),
    stats.uniform(0, 100),
    stats.uniform(0, 100),
    stats.uniform(0, 100),
    stats.uniform(0, 100),
    stats.uniform(0, 100)
]

# Prior log probability
def lnPi(theta):
    return np.sum([pi.logpdf(t) for (pi, t) in zip(priors, theta)])

# Log likelihood
def lnl(theta):
    output    = np.vstack(azr.predict(theta, dress_up=False))
    mu, y, dy = output[:, 3], output[:, 5], output[:, 6]
    lnl       = np.sum(-0.5*np.log(2*np.pi*pow(dy_bare,2)) - 0.5*pow((y - mu)/dy,2))
    return lnl

# Posterior log probability
def lnP(theta):
    theta = list( theta.valuesdict().values() )
    lnpi = lnPi(theta)
    if not np.isfinite(lnpi): return -np.inf
    return lnl(theta) + lnpi