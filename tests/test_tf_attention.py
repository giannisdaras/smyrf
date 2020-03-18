from smyrf.tf.attn import RandomBucketsAttention
import unittest
from profilehooks import timecall
import tensorflow as tf

class Profiler(unittest.TestCase):
    def test_cuda_seq_benchmark(self):
        benchmark_trials = 10
        q_seqlen = 32768
        k_seqlen = 32768
        dim = 100
        v_dim = 100

        bs = 1
        q_bucket_size = 32
        k_bucket_size = 32
        n_hashes = 16

        queries = tf.random.normal((benchmark_trials, 1, q_seqlen, dim), mean=0, stddev=1)
        keys = tf.random.normal((benchmark_trials, 1, k_seqlen, dim), mean=0, stddev=1)
        values = tf.random.normal((benchmark_trials, 1, k_seqlen, v_dim), mean=0, stddev=1)


        random_attn = RandomBucketsAttention(q_seqlen,
                                             k_seqlen,
                                             q_bucket_size=q_bucket_size,
                                             k_bucket_size=k_bucket_size,
                                             n_hashes=n_hashes,
                                             max_perms=bs)
        @timecall
        def time_profile(layer, *inputs):
            layer(*inputs)

        for i in range(benchmark_trials):
            inputs = [queries[i], keys[i], values[i]]
            time_profile(random_attn, *inputs)


if __name__ == '__main__':
    unittest.main()
