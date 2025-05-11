import logging
import os

from dag_methods import Notears, Notears_ICA_MCEM, Notears_ICA, \
                            Notears_MLP_MCEM, Notears_MLP_MCEM_INIT
from data_loader import SyntheticDataset
from miss_methods import miss_dag_gaussian, miss_dag_nongaussian, miss_dag_nonlinear
from utils.config import save_yaml_config, get_args
from utils.dir import create_dir, get_datetime_str
from utils.logging import setup_logger, get_system_info
from utils.utils import set_seed, MetricsDAG, postprocess

import numpy as np

# For logging of tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # Get arguments parsed
    args = get_args()

    # Setup for logging
    output_dir = 'output/{}'.format(get_datetime_str(add_random_str=True))
    create_dir(output_dir)  # Create directory to save log files and outputs
    setup_logger(log_path='{}/training.log'.format(output_dir), level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger.")

    # Get and save system info
    system_info = get_system_info()
    if system_info is not None:
        save_yaml_config(system_info, path='{}/system_info.yaml'.format(output_dir))

    # Save configs
    save_yaml_config(vars(args), path='{}/config.yaml'.format(output_dir))

    # Reproducibility
    set_seed(args.seed)

    # adj_mat
    prior_adj_matrix = None
    if args.prior_adj_path:
        try:
            # 假设 prior_adj_matrix[i, j] == 0 表示没有从节点 i 到节点 j 的边
            prior_adj_matrix = np.load(args.prior_adj_path)
            # 可以在这里添加对矩阵维度和d是否匹配的检查
            if prior_adj_matrix.shape != (args.d, args.d):
                _logger.warning(f"Shape of prior adjacency matrix {prior_adj_matrix.shape} "
                                f"does not match d={args.d}. Ignoring prior knowledge.")
                prior_adj_matrix = None
            else:
                _logger.info(f"Successfully loaded prior adjacency matrix from: {args.prior_adj_path}")
        except Exception as e:
            _logger.error(f"Failed to load prior adjacency matrix from {args.prior_adj_path}: {e}. "
                          f"Ignoring prior knowledge.")
            prior_adj_matrix = None

    # Load dataset
    dataset = SyntheticDataset(args.n, args.d, args.graph_type, args.degree, args.noise_type,
                               args.miss_type, args.miss_percent, args.sem_type,
                               args.equal_variances, args.mnar_type, args.p_obs, args.mnar_quantile_q)
    _logger.info("Finished loading the dataset.")

    if args.dag_method_type == 'notears':
        dag_method = Notears(args.lambda_1_ev)
    elif args.dag_method_type == 'notears_ica':
        dag_method = Notears_ICA(args.seed, args.MLEScore, prior_adj=prior_adj_matrix)
    elif args.dag_method_type == 'notears_ica_mcem':
        dag_init_method = Notears_ICA(args.seed, args.MLEScore, prior_adj=prior_adj_matrix)
        dag_method = Notears_ICA_MCEM(args.seed, args.MLEScore, prior_adj=prior_adj_matrix)
    elif args.dag_method_type == 'notears_mlp_mcem':
        dag_init_method = Notears_MLP_MCEM_INIT(prior_adj=prior_adj_matrix)
        dag_method = Notears_MLP_MCEM(prior_adj=prior_adj_matrix) 
    else:
        raise ValueError("Unknown method type.")
    _logger.info("Finished setting up the structure learning method.")

    assert args.miss_percent == 0 if args.miss_method_type == 'no_missing' \
            else args.miss_percent > 0
    
    if args.miss_method_type in {'no_missing'}:

        # Get the data for estimating DAG
        if args.miss_method_type == 'no_missing':
            X = dataset.X_true
        else:
            raise ValueError("Add your imputation methods.")

        # Estimate the DAG
        if args.dag_method_type not in {'notears_ica_mcem', 'notears_mlp_mcem'}:
            B_est = dag_method.fit(X=X, cov_emp=None, prior_adj=prior_adj_matrix)
        else:
            raise ValueError("The miss_method here does not support notears_ica_mcem/notears_mlp_mcem.")

    elif args.miss_method_type == 'miss_dag_gaussian':
        B_est, _, _ = miss_dag_gaussian(dataset.X, dataset.mask, dag_method,
                                                      args.em_iter, args.equal_variances)
    elif args.miss_method_type == 'miss_dag_nongaussian':
        assert args.dag_method_type == 'notears_ica_mcem', \
                "miss_dag_nongaussian supports only notears_ica_mcem as dag_method_type"
        B_est, _, _ = miss_dag_nongaussian(dataset.X, dag_init_method,
                                                         dag_method, args.em_iter, args.MLEScore, args.num_sampling, B_true=dataset.B_bin)
    elif args.miss_method_type == 'miss_dag_nonlinear':
        assert args.dag_method_type == 'notears_mlp_mcem', \
            "miss_dag_nonlinear supports only notears_mlp_mcem as dag_method_type"
        B_est, _, _ = miss_dag_nonlinear(dataset.X, dag_init_method,
                                                       dag_method, args.em_iter, args.equal_variances)
    else:
        raise ValueError("Unknown method type.")
    _logger.info("Finished estimating the graph.")

    # Post-process estimated solution
    _, B_processed_bin = postprocess(B_est, args.graph_thres)
    _logger.info("Finished post-processing the estimated graph.")

    raw_result = MetricsDAG(B_processed_bin, dataset.B_bin).metrics
    _logger.info("run result:{0}".format(raw_result))

if __name__ == '__main__':
    main()
