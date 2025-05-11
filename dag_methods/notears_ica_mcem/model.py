import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, n, d, num, seed=8, MLEScore='Sup-G', l1_lambda=0.1, use_float64=False, prior_adj=None): # 修改后
        self.n = n
        self.d = d
        self.num = num
        self.seed = seed
        self.MLEScore = MLEScore
        self.l1_lambda = l1_lambda
        self.tf_float_type = tf.dtypes.float64 if use_float64 else tf.dtypes.float32
        self.prior_adj_np = prior_adj # 存储 NumPy 格式的 prior_adj

        self.initializer = tf.keras.initializers.glorot_uniform(seed=self.seed)
        self._build()
        self._init_session()

    def _init_session(self):
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=0.5,
                allow_growth=True,
            )
        ))

    def _build(self):
        tf.compat.v1.reset_default_graph()

        self.rho = tf.compat.v1.placeholder(self.tf_float_type)
        self.alpha = tf.compat.v1.placeholder(self.tf_float_type)
        self.lr = tf.compat.v1.placeholder(self.tf_float_type)

        self.X = tf.compat.v1.placeholder(self.tf_float_type, shape=[self.n, self.d])
        self.weight_index = tf.compat.v1.placeholder(self.tf_float_type, shape=[self.n, 1])

        W = tf.Variable(tf.zeros([self.d, self.d], self.tf_float_type))

        if self.prior_adj_np is not None:
        # 假设 prior_adj_np[j, i] == 0 表示 W_ij (从 j 到 i 的边) 应该为 0
        # TensorFlow 中的 W 通常 W_ij 表示从 j 到 i 的边 (X_i = sum_j X_j W_ji)
        # 或者 W_ij 表示从 i 到 j 的边 (X_j = sum_i X_i W_ij)
        # 需要根据模型中 W 的实际含义来确定 prior_adj_np 的索引方式
        # 假设 W_ij 表示从 j 到 i 的边 (即 W 是转置的邻接矩阵)
        # 那么如果 prior_adj_np[source, target] == 0，则 W[target, source] 应该为0
            mask_values = np.ones((self.d, self.d), dtype=self.prior_adj_np.dtype)
            for r in range(self.d):
                for c in range(self.d):
                    # 假设 prior_adj_np[c, r] == 0 (从 c 到 r 的边不存在)
                    # 对应 W[r, c] 应该为 0
                    if self.prior_adj_np[c, r] == 0:
                        mask_values[r, c] = 0
            self.prior_mask_tf = tf.constant(mask_values, dtype=self.tf_float_type)
        else:
            self.prior_mask_tf = None

        self.W_prime = self._preprocess_graph(W)

        self.mle = self._get_mle_loss()

        self.h = tf.linalg.trace(tf.linalg.expm(self.W_prime * self.W_prime)) - self.d    # Acyclicity

        self.loss = self.mle \
                    + self.l1_lambda * tf.norm(self.W_prime, ord=1) \
                    + self.alpha * self.h + 0.5 * self.rho * self.h * self.h

        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _preprocess_graph(self, W):
        # Mask the diagonal entries of graph
        W_processed = tf.linalg.set_diag(W, tf.zeros(W.shape[0], dtype=self.tf_float_type))
        # 应用先验知识掩码
        if self.prior_mask_tf is not None:
            W_processed = W_processed * self.prior_mask_tf # 逐元素相乘
        return W_processed

    def _get_mle_loss(self):
        
        # Equal-scale version
        sigma = tf.math.sqrt(
                    tf.math.reduce_sum(
                        tf.multiply(tf.square(self.X - self.X @ self.W_prime), self.weight_index)) / (self.num * self.d )
                    )
        
        # Standardize the simulated data
        s = (self.X - self.X @ self.W_prime) / sigma
        
        if self.MLEScore == 'Sup-G':
            nm_term = 2 / self.num * tf.math.reduce_sum(
                                tf.multiply(tf.math.log(tf.math.cosh(s)), self.weight_index))
        
        elif self.MLEScore == 'Sub-G':
            nm_term = -1 / self.num * tf.math.reduce_sum(
                                tf.multiply(tf.math.log(tf.math.cosh(s)) + tf.math.square(s)/2, self.weight_index))
        else:
            raise ValueError("Unknown Score.")

        return nm_term + self.d * tf.math.log(sigma) - tf.linalg.slogdet(tf.eye(self.d) - self.W_prime)[1]