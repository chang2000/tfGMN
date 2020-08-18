from utils import *
from models import *
from datasets import *
from layers import *

import time
import random
import copy

import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


"""Evaluation"""

def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = tf.cast(tf.equal(x > 0, y > 0), dtype=tf.float32)
    return tf.reduce_mean(match, axis=1)


def compute_similarity(config, x, y):
    """Compute the distance between x and y vectors.

  The distance will be computed based on the training loss type.

  Args:
    config: a config dict.
    x: [n_examples, feature_dim] float tensor.
    y: [n_examples, feature_dim] float tensor.

  Returns:
    dist: [n_examples] float tensor.

  Raises:
    ValueError: if loss type is not supported.
  """
    if config["training"]["loss"] == "margin":
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif config["training"]["loss"] == "hamming":
        return exact_hamming_similarity(x, y)
    else:
        raise ValueError("Unknown loss type %s" % config["training"]["loss"])


def auc(scores, labels, **auc_args):
    """Compute the AUC for pair classification.

  See `tf.metrics.auc` for more details about this metric.

  Args:
    scores: [n_examples] float.  Higher scores mean higher preference of being
      assigned the label of +1.
    labels: [n_examples] int.  Labels are either +1 or -1.
    **auc_args: other arguments that can be used by `tf.metrics.auc`.

  Returns:
    auc: the area under the ROC curve.
  """
    scores_max = tf.reduce_max(scores)
    scores_min = tf.reduce_min(scores)
    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2
    # The following code should be used according to the tensorflow official
    # documentation:
    # value, _ = tf.metrics.auc(labels, scores, **auc_args)

    # However `tf.metrics.auc` is currently (as of July 23, 2019) buggy so we have
    # to use the following:
    _, value = tf.metrics.auc(labels, scores, **auc_args)
    return value

"""Build the model"""
def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

  Args:
    tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
      multiple of `n_splits`.
    n_splits: int, number of splits to split the tensor into.

  Returns:
    splits: a list of `n_splits` tensors.  The first split is [tensor[0],
      tensor[n_splits], tensor[n_splits * 2], ...], the second split is
      [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
  """
    feature_dim = tensor.shape.as_list()[-1]
    # feature dim must be known, otherwise you can provide that as an input
    assert isinstance(feature_dim, int)
    tensor = tf.reshape(tensor, [-1, feature_dim * n_splits])
    return tf.split(tensor, n_splits, axis=-1)


def build_placeholders(node_feature_dim, edge_feature_dim):
    """Build the placeholders needed for the model.

  Args:
    node_feature_dim: int.
    edge_feature_dim: int.

  Returns:
    placeholders: a placeholder name -> placeholder tensor dict.
  """
    # `n_graphs` must be specified as an integer, as `tf.dynamic_partition`
    # requires so.
    return {
        "node_features": tf.placeholder(tf.float32, [None, node_feature_dim]),
        "edge_features": tf.placeholder(tf.float32, [None, edge_feature_dim]),
        "from_idx": tf.placeholder(tf.int32, [None]),
        "to_idx": tf.placeholder(tf.int32, [None]),
        "graph_idx": tf.placeholder(tf.int32, [None]),
        # only used for pairwise training and evaluation
        "labels": tf.placeholder(tf.int32, [None]),
    }


def build_model(config, node_feature_dim, edge_feature_dim):
    """Create model for training and evaluation.

  Args:
    config: a dictionary of configs, like the one created by the
      `get_default_config` function.
    node_feature_dim: int, dimensionality of node features.
    edge_feature_dim: int, dimensionality of edge features.

  Returns:
    tensors: a (potentially nested) name => tensor dict.
    placeholders: a (potentially nested) name => tensor dict.
    model: a GraphEmbeddingNet or GraphMatchingNet instance.

  Raises:
    ValueError: if the specified model or training settings are not supported.
  """
    encoder = GraphEncoder(**config["encoder"])
    aggregator = GraphAggregator(**config["aggregator"])
    if config["model_type"] == "embedding":
        model = GraphEmbeddingNet(encoder, aggregator, **config["graph_embedding_net"])
    elif config["model_type"] == "matching":
        model = GraphMatchingNet(encoder, aggregator, **config["graph_matching_net"])
    else:
        raise ValueError("Unknown model type: %s" % config["model_type"])

    training_n_graphs_in_batch = config["training"]["batch_size"]
    if config["training"]["mode"] == "pair":
        training_n_graphs_in_batch *= 2
    elif config["training"]["mode"] == "triplet":
        training_n_graphs_in_batch *= 4
    else:
        raise ValueError("Unknown training mode: %s" % config["training"]["mode"])

    placeholders = build_placeholders(node_feature_dim, edge_feature_dim)

    # training
    model_inputs = placeholders.copy()
    del model_inputs["labels"]
    model_inputs["n_graphs"] = training_n_graphs_in_batch
    graph_vectors = model(**model_inputs)

    if config["training"]["mode"] == "pair":
        x, y = reshape_and_split_tensor(graph_vectors, 2)
        loss = pairwise_loss(
            x,
            y,
            placeholders["labels"],
            loss_type=config["training"]["loss"],
            margin=config["training"]["margin"],
        )

        # optionally monitor the similarity between positive and negative pairs
        is_pos = tf.cast(tf.equal(placeholders["labels"], 1), tf.float32)
        is_neg = 1 - is_pos
        n_pos = tf.reduce_sum(is_pos)
        n_neg = tf.reduce_sum(is_neg)
        sim = compute_similarity(config, x, y)
        sim_pos = tf.reduce_sum(sim * is_pos) / (n_pos + 1e-8)
        sim_neg = tf.reduce_sum(sim * is_neg) / (n_neg + 1e-8)
    else:
        x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
        loss = triplet_loss(
            x_1,
            y,
            x_2,
            z,
            loss_type=config["training"]["loss"],
            margin=config["training"]["margin"],
        )

        sim_pos = tf.reduce_mean(compute_similarity(config, x_1, y))
        sim_neg = tf.reduce_mean(compute_similarity(config, x_2, z))

    graph_vec_scale = tf.reduce_mean(graph_vectors ** 2)
    if config["training"]["graph_vec_regularizer_weight"] > 0:
        loss += (
            config["training"]["graph_vec_regularizer_weight"] * 0.5 * graph_vec_scale
        )

    # monitor scale of the parameters and gradients, these are typically helpful
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config["training"]["learning_rate"]
    )
    grads_and_params = optimizer.compute_gradients(loss)
    grads, params = zip(*grads_and_params)
    grads, _ = tf.clip_by_global_norm(grads, config["training"]["clip_value"])
    train_step = optimizer.apply_gradients(zip(grads, params))

    grad_scale = tf.global_norm(grads)
    param_scale = tf.global_norm(params)

    # evaluation
    model_inputs["n_graphs"] = config["evaluation"]["batch_size"] * 2
    eval_pairs = model(**model_inputs)
    x, y = reshape_and_split_tensor(eval_pairs, 2)
    similarity = compute_similarity(config, x, y)
    pair_auc = auc(similarity, placeholders["labels"])

    model_inputs["n_graphs"] = config["evaluation"]["batch_size"] * 4
    eval_triplets = model(**model_inputs)
    x_1, y, x_2, z = reshape_and_split_tensor(eval_triplets, 4)
    sim_1 = compute_similarity(config, x_1, y)
    sim_2 = compute_similarity(config, x_2, z)
    triplet_acc = tf.reduce_mean(tf.cast(sim_1 > sim_2, dtype=tf.float32))

    return (
        {
            "train_step": train_step,
            "metrics": {
                "training": {
                    "loss": loss,
                    "grad_scale": grad_scale,
                    "param_scale": param_scale,
                    "graph_vec_scale": graph_vec_scale,
                    "sim_pos": sim_pos,
                    "sim_neg": sim_neg,
                    "sim_diff": sim_pos - sim_neg,
                },
                "validation": {"pair_auc": pair_auc, "triplet_acc": triplet_acc,},
            },
        },
        placeholders,
        model,
    )

"""Training Pipeline"""
def build_datasets(config):
    """Build the training and evaluation datasets."""
    config = copy.deepcopy(config)

    if config["data"]["problem"] == "graph_edit_distance":
        dataset_params = config["data"]["dataset_params"]
        validation_dataset_size = dataset_params["validation_dataset_size"]
        del dataset_params["validation_dataset_size"]
        training_set = GraphEditDistanceDataset(**dataset_params)
        dataset_params["dataset_size"] = validation_dataset_size
        validation_set = FixedGraphEditDistanceDataset(**dataset_params)
    else:
        raise ValueError("Unknown problem type: %s" % config["data"]["problem"])
    return training_set, validation_set


def fill_feed_dict(placeholders, batch):
    """Create a feed dict for the given batch of data.

  Args:
    placeholders: a dict of placeholders.
    batch: a batch of data, should be either a single `GraphData` instance for
      triplet training, or a tuple of (graphs, labels) for pairwise training.

  Returns:
    feed_dict: a feed_dict that can be used in a session run call.
  """
    if isinstance(batch, GraphData):
        graphs = batch
        labels = None
    else:
        graphs, labels = batch

    feed_dict = {
        placeholders["node_features"]: graphs.node_features,
        placeholders["edge_features"]: graphs.edge_features,
        placeholders["from_idx"]: graphs.from_idx,
        placeholders["to_idx"]: graphs.to_idx,
        placeholders["graph_idx"]: graphs.graph_idx,
    }
    if labels is not None:
        feed_dict[placeholders["labels"]] = labels
    return feed_dict


def evaluate(sess, eval_metrics, placeholders, validation_set, batch_size):
    """Evaluate model performance on the given validation set.

  Args:
    sess: a `tf.Session` instance used to run the computation.
    eval_metrics: a dict containing two tensors 'pair_auc' and 'triplet_acc'.
    placeholders: a placeholder dict.
    validation_set: a `GraphSimilarityDataset` instance, calling `pairs` and
      `triplets` functions with `batch_size` creates iterators over a finite
      sequence of batches to evaluate on.
    batch_size: number of batches to use for each session run call.

  Returns:
    metrics: a dict of metric name => value mapping.
  """
    accumulated_pair_auc = []
    for batch in validation_set.pairs(batch_size):
        feed_dict = fill_feed_dict(placeholders, batch)
        pair_auc = sess.run(eval_metrics["pair_auc"], feed_dict=feed_dict)
        accumulated_pair_auc.append(pair_auc)

    accumulated_triplet_acc = []
    for batch in validation_set.triplets(batch_size):
        feed_dict = fill_feed_dict(placeholders, batch)
        triplet_acc = sess.run(eval_metrics["triplet_acc"], feed_dict=feed_dict)
        accumulated_triplet_acc.append(triplet_acc)

    return {
        "pair_auc": np.mean(accumulated_pair_auc),
        "triplet_acc": np.mean(accumulated_triplet_acc),
    }

"""Main run process"""

config = get_default_config()
config["training"]["n_training_steps"] = 5000
tf.reset_default_graph()

# Set random seeds
seed = config["seed"]
random.seed(seed)
np.random.seed(seed + 1)
tf.set_random_seed(seed + 2)

training_set, validation_set = build_datasets(config)



if config["training"]["mode"] == "pair":
    training_data_iter = training_set.pairs(config["training"]["batch_size"])
    first_batch_graphs, _ = next(training_data_iter)
else:
    training_data_iter = training_set.triplets(config["training"]["batch_size"])
    first_batch_graphs = next(training_data_iter)

node_feature_dim = first_batch_graphs.node_features.shape[-1]
edge_feature_dim = first_batch_graphs.edge_features.shape[-1]

tensors, placeholders, model = build_model(config, node_feature_dim, edge_feature_dim)

accumulated_metrics = collections.defaultdict(list)

t_start = time.time()

init_ops = (tf.global_variables_initializer(), tf.local_variables_initializer())

# If we already have a session instance, close it and start a new one
if "sess" in globals():
    sess.close()

# We will need to keep this session instance around for e.g. visualization.
# But you should probably wrap it in a `with tf.Session() sess:` context if you
# want to use the code elsewhere.
sess = tf.Session()
sess.run(init_ops)

# use xrange here if you are still on python 2
for i_iter in range(config["training"]["n_training_steps"]):
    batch = next(training_data_iter)
    _, train_metrics = sess.run(
        [tensors["train_step"], tensors["metrics"]["training"]],
        feed_dict=fill_feed_dict(placeholders, batch),
    )

    # accumulate over minibatches to reduce variance in the training metrics
    for k, v in train_metrics.items():
        accumulated_metrics[k].append(v)

    if (i_iter + 1) % config["training"]["print_after"] == 0:
        metrics_to_print = {k: np.mean(v) for k, v in accumulated_metrics.items()}
        info_str = ", ".join(["%s %.4f" % (k, v) for k, v in metrics_to_print.items()])
        # reset the metrics
        accumulated_metrics = collections.defaultdict(list)

        if (i_iter + 1) // config["training"]["print_after"] % config["training"][
            "eval_after"
        ] == 0:
            eval_metrics = evaluate(
                sess,
                tensors["metrics"]["validation"],
                placeholders,
                validation_set,
                config["evaluation"]["batch_size"],
            )
            info_str += ", " + ", ".join(
                ["%s %.4f" % ("val/" + k, v) for k, v in eval_metrics.items()]
            )

        print("iter %d, %s, time %.2fs" % (i_iter + 1, info_str, time.time() - t_start))
        t_start = time.time()


# Note that albeit a bit noisy, the loss is going down, the similarity gap
# between positive and negative pairs are growing and the evaluation results, i.e. pair AUC and triplet accuracies are going up as well.  Overall training seems to be working!
#
# You can train this much longer.  We observed improvement in performance even after training for 500,000 steps, but didn't push this much further as it is a synthetic task after all.

# ## Test the model and create some visualizations
#
# Once the model is trained, we can test in on unseen data.  Our graph matching networks use cross-graph matching-based attention to compute graph similarity, we can visualize these attention weights to see where the model is attending to.


# visualize on graphs of 10 nodes, bigger graphs become more difficult to
# visualize
vis_dataset = GraphEditDistanceDataset([10, 10], [0.2, 0.2], 1, 2, permute=False)

pair_iter = vis_dataset.pairs(2)
graphs, labels = next(pair_iter)


# Let's split the batched graphs into individual graphs and visualize them first.

def split_graphs(graphs):
    """Split a batch of graphs into individual `nx.Graph` instances."""
    g = [nx.Graph() for _ in range(graphs.n_graphs)]
    node_ids = np.arange(graphs.graph_idx.size, dtype=np.int32)
    for i in range(graphs.n_graphs):
        nodes_in_graph = node_ids[graphs.graph_idx == i]
        n_nodes = len(nodes_in_graph)
        g[i].add_nodes_from(range(n_nodes))
        node_id_min = nodes_in_graph.min()
        node_id_max = nodes_in_graph.max()

        edges = []
        for u, v in zip(graphs.from_idx, graphs.to_idx):
            if node_id_min <= u <= node_id_max and node_id_min <= v <= node_id_max:
                edges.append((u - node_id_min, v - node_id_min))
        g[i].add_edges_from(edges)

    return g


nx_graphs = split_graphs(graphs)

for i in range(0, len(nx_graphs), 2):
    label = labels[i // 2]
    g1 = nx_graphs[i]
    g2 = nx_graphs[i + 1]
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(121)
    # Compute the positions of graphs first to make sure the visualizations are
    # consistent.
    pos = nx.drawing.spring_layout(g1)
    nx.draw_networkx(g1, pos=pos, ax=ax)
    ax.set_title("Graph 1")
    ax.axis("off")
    ax = plt.subplot(122)
    nx.draw_networkx(g2, pos=pos, ax=ax)
    ax.set_title("Graph 2")
    ax.axis("off")


# Build the computation graph for visualization.

n_graphs = graphs.n_graphs

model_inputs = placeholders.copy()
del model_inputs["labels"]
graph_vectors = model(n_graphs=n_graphs, **model_inputs)
x, y = reshape_and_split_tensor(graph_vectors, 2)
similarity = compute_similarity(config, x, y)

layer_outputs = model.get_layer_outputs()



def build_matchings(layer_outputs, graph_idx, n_graphs, sim):
    """Build the matching attention matrices from layer outputs."""
    assert n_graphs % 2 == 0
    attention = []
    for h in layer_outputs:
        partitions = tf.dynamic_partition(h, graph_idx, n_graphs)
        attention_in_layer = []
        for i in range(0, n_graphs, 2):
            x = partitions[i]
            y = partitions[i + 1]
            a = sim(x, y)
            a_x = tf.nn.softmax(a, axis=1)  # i->j
            a_y = tf.nn.softmax(a, axis=0)  # j->i
            attention_in_layer.append((a_x, a_y))
        attention.append(attention_in_layer)
    return attention


attentions = build_matchings(
    layer_outputs,
    placeholders["graph_idx"],
    n_graphs,
    get_pairwise_similarity(config["graph_matching_net"]["similarity"]),
)

sim, a = sess.run(
    [similarity, attentions], feed_dict=fill_feed_dict(placeholders, (graphs, labels))
)


print(labels)
print(sim)

# Similarity for positive pair is much higher than the similarity for the negative pair.
#
# Remember that with a margin loss and Euclidean distance, which is how this model is trained, the similarity value is the negative distance.  In this case the distance for positive pair is quite small, while the distance between two graphs in the negative pair is large.

