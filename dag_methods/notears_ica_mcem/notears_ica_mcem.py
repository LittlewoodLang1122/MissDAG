from dag_methods.notears_ica_mcem.model import Model
from dag_methods.notears_ica_mcem.al_trainer import ALTrainer
import tensorflow as tf
import logging

class Notears_ICA_MCEM:
    _logger = logging.getLogger(__name__)

class Notears_ICA_MCEM:
    _logger = logging.getLogger(__name__)

    # def __init__(self, seed=2021, MLEScore='Sup-G', use_float64=False): # 原来的
    def __init__(self, seed=2021, MLEScore='Sup-G', use_float64=False, prior_adj=None): # 修改后
        self.seed = seed
        self.MLEScore = MLEScore
        self.tf_float_type = tf.dtypes.float64 if use_float64 else tf.dtypes.float32
        self.prior_adj = prior_adj # 存储先验知识

    def fit(self, X_sampling, num, weight_index):
        n, d = X_sampling.shape
        # 将 prior_adj 传递给 Model
        model = Model(n, d, num, seed=self.seed, MLEScore=self.MLEScore, prior_adj=self.prior_adj, use_float64=(self.tf_float_type == tf.dtypes.float64))
        trainer = ALTrainer(init_rho=1.0, rho_max=1e16, h_factor=0.25, rho_multiply=10.0,
                            init_iter=3, learning_rate=1e-3, h_tol=1e-8)
        W_est = trainer.train(model, X_sampling, weight_index, max_iter=20, iter_step=1500)
        return W_est

