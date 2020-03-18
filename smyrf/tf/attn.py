import tensorflow as tf

def sort_key_val(t1, t2, dim=-1, n_buckets=1):
    # TODO (@giannisdaras): do that with one sort
    values = tf.sort(t1, axis=dim)
    indices = tf.argsort(t1, axis=dim)
    t2 = tf.broadcast_to(t2, tf.shape(t1))
    gathered = tf.squeeze(tf.gather(t2, indices, axis=dim))
    batch_size = tf.shape(t2)[0]
    N = tf.shape(t2)[1]
    return values, tf.broadcast_to(gathered, (batch_size, N))



def batched_index_select(values, indices):
    batch_size = tf.shape(values)[0]
    N = tf.shape(indices)[1]
    last_dim = tf.shape(values)[-1]
    gathered = tf.squeeze(tf.gather(values, indices[:, :, None], axis=1))
    return tf.broadcast_to(gathered, (batch_size, N, last_dim))


class RandomBucketsAttention():
    def __init__(self,
                 q_seqlen,
                 k_seqlen,
                 dropout=0.,
                 q_bucket_size=64,
                 k_bucket_size=64,
                 n_hashes=8,
                 max_perms=20,
                 add_local_attn_hash=False,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 drop_for_hash_rate=0.0,
                 return_attn=False):
        '''
            Args:
                - q_seqlen: query seq length
                - k_seqlen
                - dropout
                - q_bucket_size
                - k_bucket_size
                - n_hashes
                - add_local_attn_hash
                - causal: If true, constraints attention only to the left side
                - allow_duplicate_attention
                - attend_across_buckets
                - rehash_each_round
                - drop_for_hash_rate
                - return_attn: Whether to return attention map
                - max_perms: maximum batch size (needed for speed)
                - device
        '''
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = lambda x: tf.nn.dropout(x, dropout)
        self.dropout_for_hash = lambda x: tf.nn.dropout(x, drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

        self.n_hashes = n_hashes
        self.add_local_attn_hash = add_local_attn_hash

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # maximum number of pre-computed permutations
        self.max_perms = max_perms

        # needed for pre-computing permutations
        self.q_seqlen = q_seqlen
        self.k_seqlen = k_seqlen


        # pre-computed for speed

        q_indices = tf.broadcast_to(tf.range(0, q_seqlen), (max_perms, n_hashes, q_seqlen))
        k_indices = tf.broadcast_to(tf.range(0, k_seqlen), (max_perms, n_hashes, k_seqlen))

        q_random_perms = tf.transpose(tf.random.shuffle(tf.transpose(q_indices, perm=(2, 0, 1))), perm=(1, 2, 0))
        k_random_perms = tf.transpose(tf.random.shuffle(tf.transpose(k_indices, perm=(2, 0, 1))), perm=(1, 2, 0))


        hashes_offset = tf.fill((n_hashes,), q_seqlen) * tf.range(0, n_hashes)

        self.s_q_ticker = tf.transpose(q_random_perms, perm=(0, 2, 1)) + hashes_offset
        self.s_k_ticker = tf.transpose(k_random_perms, perm=(0, 2, 1)) + hashes_offset

        del q_random_perms
        del k_random_perms
        del hashes_offset

        self.s_q_ticker = tf.reshape(tf.transpose(self.s_q_ticker, perm=(0, 2, 1)), (max_perms, -1))
        self.s_k_ticker = tf.reshape(tf.transpose(self.s_k_ticker, perm=(0, 2, 1)), (max_perms, -1))

        q_ticker = tf.broadcast_to(tf.expand_dims(tf.range(0, n_hashes * q_seqlen), 0), tf.shape(self.s_q_ticker))
        k_ticker = tf.broadcast_to(tf.expand_dims(tf.range(0, n_hashes * k_seqlen), 0), tf.shape(self.s_k_ticker))

        _, self.q_undo_sort = sort_key_val(self.s_q_ticker, q_ticker, dim=-1)
        _, self.k_undo_sort = sort_key_val(self.s_k_ticker, k_ticker, dim=-1)

        del q_ticker
        del k_ticker


    def __call__(self, query, key, value, input_mask=None, input_attn_mask=None):
        batch_size, q_seqlen, dim = tf.shape(query)
        batch_size, k_seqlen, dim = tf.shape(key)
        batch_size, k_seqlen, v_dim = tf.shape(value)


        # pre-compute more random perms than batch size
        assert batch_size <= self.max_perms


        # we need the same number of buckets for queries and keys
        assert (q_seqlen // self.q_bucket_size) == (k_seqlen // self.k_bucket_size)
        n_buckets = q_seqlen // self.q_bucket_size

        s_q_ticker = self.s_q_ticker[:batch_size]
        s_k_ticker = self.s_k_ticker[:batch_size]

        q_undo_sort = self.q_undo_sort[:batch_size]
        k_undo_sort = self.k_undo_sort[:batch_size]

        # fix range
        q_st = s_q_ticker % q_seqlen
        k_st = s_k_ticker % k_seqlen

        sq = batched_index_select(query, q_st)
        sk = batched_index_select(key, k_st)
        sv = batched_index_select(value, k_st)


        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = self.n_hashes * n_buckets

        # reshape tickers
        bq_t = tf.reshape(q_st, (batch_size, chunk_size, -1))
        bk_t = tf.reshape(k_st, (batch_size, chunk_size, -1))

        # reshape arrays
        bq = tf.reshape(sq, (batch_size, chunk_size, -1, dim))
        bk = tf.reshape(sk, (batch_size, chunk_size, -1, dim))
        bv = tf.reshape(sv, (batch_size, chunk_size, -1, v_dim))

        # TODO(giannisdaras): allow lookback attention

        # Dot-product attention.
        inner = tf.einsum('bhie,bhje->bhij', bq, bk)

        # TODO(giannisdaras): allow masking

        # TODO(giannidaras): allow attention between buckets

        # TODO(giannisdaras): allow duplicate attention

        # Softmax.
        dots_logsumexp = tf.math.reduce_logsumexp(inner, axis=-1, keepdims=True)
        dots = tf.math.exp(inner - dots_logsumexp)
        dropped_dots = self.dropout(dots)

        bo = tf.einsum('buij,buje->buie', dropped_dots, bv)
        so = tf.reshape(bo, (batch_size, -1, v_dim))
        slogits = tf.reshape(dots_logsumexp, (batch_size, -1,))

        # re-arrange
        o = batched_index_select(so, q_undo_sort)
        logits = tf.gather(slogits, q_undo_sort, axis=-1)


        o = tf.reshape(o, (batch_size, self.n_hashes, q_seqlen, -1))
        # ln(total mass of softmax for each query)
        logits = tf.reshape(logits, (batch_size, self.n_hashes, q_seqlen, 1))


        probs = tf.math.exp(logits - tf.math.reduce_logsumexp(logits, axis=1, keepdims=True))
        out = tf.math.reduce_sum(o * probs, axis=1)


        return out
