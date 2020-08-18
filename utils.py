import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.

  This function computes the following similarity value between each pair of x_i
  and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

  Args:
    x: NxD float tensor.
    y: MxD float tensor.

  Returns:
    s: NxM float tensor, the pairwise euclidean similarity.
  """
    s = 2 * tf.matmul(x, y, transpose_b=True)
    diag_x = tf.reduce_sum(x * x, axis=-1, keepdims=True)
    diag_y = tf.reshape(tf.reduce_sum(y * y, axis=-1), (1, -1))
    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    """Compute the dot product similarity between x and y.

  This function computes the following similarity value between each pair of x_i
  and y_j: s(x_i, y_j) = x_i^T y_j.

  Args:
    x: NxD float tensor.
    y: MxD float tensor.

  Returns:
    s: NxM float tensor, the pairwise dot product similarity.
  """
    return tf.matmul(x, y, transpose_b=True)


def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.

  This function computes the following similarity value between each pair of x_i
  and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

  Args:
    x: NxD float tensor.
    y: MxD float tensor.

  Returns:
    s: NxM float tensor, the pairwise cosine similarity.
  """
    x = tf.nn.l2_normalize(x, axis=-1)
    y = tf.nn.l2_normalize(y, axis=-1)
    return tf.matmul(x, y, transpose_b=True)


PAIRWISE_SIMILARITY_FUNCTION = {
    "euclidean": pairwise_euclidean_similarity,
    "dotproduct": pairwise_dot_product_similarity,
    "cosine": pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
    """Get pairwise similarity metric by name.

  Args:
    name: string, name of the similarity metric, one of {dot-product, cosine,
      euclidean}.

  Returns:
    similarity: a (x, y) -> sim function.

  Raises:
    ValueError: if name is not supported.
  """
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]

"""Cross Graph Attention"""

def compute_cross_attention(x, y, sim):
    """Compute cross attention.

  x_i attend to y_j:
  a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
  y_j attend to x_i:
  a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
  attention_x = sum_j a_{i->j} y_j
  attention_y = sum_i a_{j->i} x_i

  Args:
    x: NxD float tensor.
    y: MxD float tensor.
    sim: a (x, y) -> similarity function.

  Returns:
    attention_x: NxD float tensor.
    attention_y: NxD float tensor.
  """
    a = sim(x, y)
    a_x = tf.nn.softmax(a, axis=1)  # i->j
    a_y = tf.nn.softmax(a, axis=0)  # j->i
    attention_x = tf.matmul(a_x, y)
    attention_y = tf.matmul(a_y, x, transpose_a=True)
    return attention_x, attention_y


def batch_block_pair_attention(data, block_idx, n_blocks, similarity="dotproduct"):
    """Compute batched attention between pairs of blocks.

  This function partitions the batch data into blocks according to block_idx.
  For each pair of blocks, x = data[block_idx == 2i], and
  y = data[block_idx == 2i+1], we compute

  x_i attend to y_j:
  a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
  y_j attend to x_i:
  a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

  and

  attention_x = sum_j a_{i->j} y_j
  attention_y = sum_i a_{j->i} x_i.

  Args:
    data: NxD float tensor.
    block_idx: N-dim int tensor.
    n_blocks: integer.
    similarity: a string, the similarity metric.

  Returns:
    attention_output: NxD float tensor, each x_i replaced by attention_x_i.

  Raises:
    ValueError: if n_blocks is not an integer or not a multiple of 2.
  """
    if not isinstance(n_blocks, int):
        raise ValueError("n_blocks (%s) has to be an integer." % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError("n_blocks (%d) must be a multiple of 2." % n_blocks)

    # Equation 9
    sim = get_pairwise_similarity(similarity)

    results = []
    partitions = tf.dynamic_partition(data, block_idx, n_blocks)

    # It is rather complicated to allow n_blocks be a tf tensor and do this in a
    # dynamic loop, and probably unnecessary to do so.  Therefore we are
    # restricting n_blocks to be a integer constant here and using the plain for
    # loop.
    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y, sim)
        results.append(attention_x)
        results.append(attention_y)

    results = tf.concat(results, axis=0)
    # the shape of the first dimension is lost after concat, reset it back
    results.set_shape(data.shape)
    return results

"""Training on Pairs"""

def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return tf.reduce_sum((x - y) ** 2, axis=-1)

def approximate_hamming_similarity(x, y):
    """Approximate Hamming similarity."""
    return tf.reduce_mean(tf.tanh(x) * tf.tanh(y), axis=1)

def pairwise_loss(x, y, labels, loss_type="margin", margin=1.0):
    """Compute pairwise loss.

  Args:
    x: [N, D] float tensor, representations for N examples.
    y: [N, D] float tensor, representations for another N examples.
    labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
      and y[i] are similar, and -1 otherwise.
    loss_type: margin or hamming.
    margin: float scalar, margin for the margin loss.

  Returns:
    loss: [N] float tensor.  Loss for each pair of representations.
  """
    labels = tf.cast(labels, x.dtype)
    if loss_type == "margin":
        return tf.nn.relu(margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == "hamming":
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError("Unknown loss_type %s" % loss_type)

"""Training on Triplets"""

def triplet_loss(x_1, y, x_2, z, loss_type="margin", margin=1.0):
    """Compute triplet loss.

  This function computes loss on a triplet of inputs (x, y, z).  A similarity or
  distance value is computed for each pair of (x, y) and (x, z).  Since the
  representations for x can be different in the two pairs (like our matching
  model) we distinguish the two x representations by x_1 and x_2.

  Args:
    x_1: [N, D] float tensor.
    y: [N, D] float tensor.
    x_2: [N, D] float tensor.
    z: [N, D] float tensor.
    loss_type: margin or hamming.
    margin: float scalar, margin for the margin loss.

  Returns:
    loss: [N] float tensor.  Loss for each pair of representations.
  """
    if loss_type == "margin":
        return tf.nn.relu(
            margin + euclidean_distance(x_1, y) - euclidean_distance(x_2, z)
        )
    elif loss_type == "hamming":
        return 0.125 * (
            (approximate_hamming_similarity(x_1, y) - 1) ** 2
            + (approximate_hamming_similarity(x_2, z) + 1) ** 2
        )
    else:
        raise ValueError("Unknown loss_type %s" % loss_type)

"""Configs"""
def get_default_config():
    """The default configs."""
    node_state_dim = 32
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here.
        node_update_type="gru",
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"

    return dict(
        encoder=dict(node_hidden_sizes=[node_state_dim], edge_hidden_sizes=None),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            gated=True,
            aggregation_type="sum",
        ),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        # Set to `embedding` to use the graph embedding net.
        model_type="matching",
        data=dict(
            problem="graph_edit_distance",
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                n_nodes_range=[20, 20],
                p_edge_range=[0.2, 0.2],
                n_changes_positive=1,
                n_changes_negative=2,
                validation_dataset_size=1000,
            ),
        ),
        training=dict(
            batch_size=20,
            learning_rate=1e-3,
            mode="pair",
            loss="margin",
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            n_training_steps=10000,
            # Print training information every this many training steps.
            print_after=100,
            # Evaluate on validation set every `eval_after * print_after` steps.
            eval_after=10,
        ),
        evaluation=dict(batch_size=20),
        seed=8,
    )

