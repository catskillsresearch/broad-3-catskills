import os, luigi, torch
from ridge_regression import *
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from np_loadz import np_loadz
    
class template_ridge_transform(luigi.Task):
    src_task = luigi.TaskParameter()
    fit_task = luigi.TaskParameter()
    tgt_object_type = luigi.Parameter()
    tgt_object_name = luigi.Parameter()
    
    def requires(self):
        return {'src': self.src_task(), 'fit': self.fit_task()}
        
    def output(self):
        map_name = f'{self.tgt_object_name}_{self.tgt_object_type}'
        return luigi.LocalTarget(f'resources/run/{map_name}.npz')
       
    def run(self):
        X = np_loadz(self.input()['src']['PCs'].path)
        W = np_loadz(self.input()['fit']['W'].path)
        Y_hat = ridge_apply(X, W)
        np.savez_compressed(self.output().path, Y_hat)
