import bilby

import numpy as np

from pyazr import azure2
from multiprocessing import Pool, current_process

class Likelihood(bilby.Likelihood):

    def __init__(self, file, nprocs):
        self.azr = azure2(file, nprocs=nprocs)
        self.ntheta = len(self.azr.params)

        # We'll read the data from the output file since it's already in the center-of-mass frame
        self.y = self.azr.cross
        self.yerr = self.azr.cross_err
        self.ndata = sum( len(segment) for segment in self.y )

        theta0 = self.azr.params 
        norms = [1.0 for _ in range( len(self.y) )]
        theta0 = np.concatenate( (theta0, norms) )

        super().__init__(parameters={"param_{}".format(i): theta0[i] for i in range( len( theta0 ) ) })

    def log_likelihood( self ):
        res = 0
        try: proc = int(current_process()._identity[0] - 1)
        except: proc = 0
        theta = np.array( [self.parameters["param_{}".format(i)] for i in range( len( self.parameters ) )] )
        mu = self.azr.calculate( theta[:self.ntheta], proc=proc )
        for i in range( len( mu ) ):
            idx = self.ntheta + i
            res += np.sum( -0.5 * np.log(2 * np.pi * pow(self.yerr[i], 2) ) - 0.5 * pow((mu[i] - self.y[i] * theta[idx]) / (self.yerr[i] * theta[idx]), 2) )
        return res