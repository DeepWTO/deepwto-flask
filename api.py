# -*- coding:utf-8 -*-
__author__ = "Randolph"
__modify__ = "Zachary"

import os
import sys
import time
import logging
import tensorflow as tf

from model import OneLabelTextCNN
import feed
import utils
import constants

# Parameters
# ==================================================

TRAIN_OR_RESTORE = input("☛ Train or Restore?(T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ["T", "R"]):
    TRAIN_OR_RESTORE = input(
        "✘ The format of your input is illegal, " "please re-input: "
    )
logging.info("✔︎ The format of your input is legal, " "now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == "T":
    logger = utils.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()))
if TRAIN_OR_RESTORE == "R":
    logger = utils.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()))

VALIDATIONSET_DIR = "./data/test_data.json"
METADATA_DIR = "../data/metadata.tsv"

# Data Parameters

tf.flags.DEFINE_string(
    "validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data."
)

tf.flags.DEFINE_string(
    "metadata_file",
    METADATA_DIR,
    "Metadata file for embedding visualization"
    "(Each line is a word segment in metadata_file).",
)

tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")

# Model Hyperparameters
tf.flags.DEFINE_float("pos_weight", 100000, "Coefficient To prevent False Positive")

tf.flags.DEFINE_float("learning_rate", 0.01, "The learning rate (default: 0.001)")

tf.flags.DEFINE_integer(
    "pad_seq_len_gov",
    35842,
    "Recommended padding Sequence length of data for "
    "gov measure description "
    "(depends on the data)",
)

tf.flags.DEFINE_integer(
    "pad_seq_len_art",
    20158,
    "Recommended padding Sequence length of data for "
    "provision(article) text"
    "(depends on the data)",
)

tf.flags.DEFINE_integer(
    "embedding_dim", 300, "Dimensionality of character embedding (default: 128)"
)

tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")

tf.flags.DEFINE_integer(
    "fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)"
)

tf.flags.DEFINE_string(
    "filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')"
)

tf.flags.DEFINE_integer(
    "num_filters", 128, "Number of filters per filter size (default: 128)"
)

tf.flags.DEFINE_float(
    "dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)"
)

tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_integer("num_classes", 1, "Number of labels (depends on the task)")

tf.flags.DEFINE_integer("top_num", 1, "Number of top K prediction classes (default: 5)")

tf.flags.DEFINE_float(
    "threshold", 0.5, "Threshold for prediction classes (default: 0.5)"
)

# Training Parameters
tf.flags.DEFINE_integer(
    "batch_size",
    1,
    # 8 for zacbuntu
    "Batch Size (default: 256)",
)

tf.flags.DEFINE_integer(
    "num_epochs", 10000000000, "Number of training epochs (default: 100)"
)

tf.flags.DEFINE_integer(
    "evaluate_every",
    30,
    "Evaluate model on dev set after this many steps " "(default: 5000)",
)

tf.flags.DEFINE_float(
    "norm_ratio",
    1,
    "The ratio of the sum of gradients norms of trainable " "variable (default: 1.25)",
)

tf.flags.DEFINE_integer(
    "decay_steps", 5000, "how many steps before decay learning rate. " "(default: 500)"
)

tf.flags.DEFINE_float(
    "decay_rate", 0.95, "Rate of decay for learning rate. (default: 0.95)"
)

tf.flags.DEFINE_integer(
    "checkpoint_every", 60, "Save model after this many steps (default: 1000)"
)

tf.flags.DEFINE_integer(
    "num_checkpoints", 3, "Number of checkpoints to store (default: 50)"
)

# Misc Parameters
tf.flags.DEFINE_boolean(
    "allow_soft_placement", True, "Allow device soft device placement"
)
tf.flags.DEFINE_boolean(
    "log_device_placement", False, "Log placement of ops on devices"
)
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = "-" * 100
logger.info(
    "\n".join(
        [
            dilim,
            *[
                "{0:>50}|{1:<50}".format(attr.upper(), FLAGS.__getattr__(attr))
                for attr in sorted(FLAGS.__dict__["__wrapped"])
            ],
            dilim,
        ]
    )
)


def test(word2vec_path):
    """Training TextCNN-one-label model."""

    # Load sentences, labels, and training parameters
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Validation data processing...")
    val_data = feed.load_data_and_labels_one_label(
        FLAGS.validation_data_file,
        FLAGS.num_classes,
        FLAGS.embedding_dim,
        word2vec_path=word2vec_path
    )

    logger.info(
        "Recommended padding Sequence length for GOV_MEASURE is: "
        "{}".format(FLAGS.pad_seq_len_gov)
    )

    logger.info(
        "Recommended padding Sequence length for Article is: "
        "{}".format(FLAGS.pad_seq_len_art)
    )

    logger.info("✔︎ GOV MEASURE padding...")

    logger.info("✔︎ Validation data padding...")
    x_val_gov, x_val_art, y_val = feed.pad_data_one_label(
        val_data, FLAGS.pad_seq_len_gov, FLAGS.pad_seq_len_art
    )

    x_val_testid = val_data.testid

    print("x_val_gov: ", len(x_val_gov))
    print("x_val_testid", len(x_val_testid))
    print("x_val_art: ", len(x_val_art))

    # Build vocabulary

    VOCAB_SIZE = feed.load_vocab_size(word2vec_path=word2vec_path)

    # Use pretrained W2V
    pretrained_word2vec_matrix = feed.load_word2vec_matrix(
        VOCAB_SIZE, FLAGS.embedding_dim, word2vec_path=word2vec_path
    )

    # Build a graph and cnn object
    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
        )

        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = OneLabelTextCNN(
                sequence_length_gov=FLAGS.pad_seq_len_gov,
                sequence_length_art=FLAGS.pad_seq_len_art,
                vocab_size=VOCAB_SIZE,
                fc_hidden_size=FLAGS.fc_hidden_size,
                embedding_size=FLAGS.embedding_dim,
                embedding_type=FLAGS.embedding_type,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pretrained_embedding=pretrained_word2vec_matrix,
            )

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(
                    learning_rate=FLAGS.learning_rate,
                    global_step=cnn.global_step,
                    decay_steps=FLAGS.decay_steps,
                    decay_rate=FLAGS.decay_rate,
                    staircase=True,
                )
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, variables = zip(*optimizer.compute_gradients(cnn.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, variables):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        "{0}/grad/hist".format(v.name), g
                    )
                    sparsity_summary = tf.summary.scalar(
                        "{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g)
                    )
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if FLAGS.train_or_restore == "R":
                MODEL = input(
                    "☛ Please input the checkpoints model you want "
                    "to restore, it should be like(1490175368): "
                )
                # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input(
                        "✘ The format of your input is illegal, " "please re-input: "
                    )
                logger.info(
                    "✔︎ The format of your input is legal, "
                    "now loading to next step..."
                )
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("✔︎ Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(
                    os.path.join(os.path.curdir, "runs", timestamp)
                )
                logger.info("✔︎ Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

            if FLAGS.train_or_restore == "R":
                # Load cnn model
                logger.info("✔︎ Loading model...")
                print(checkpoint_dir)
                # checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                checkpoint_file = "/Users/zachary/deepwto/deepwto-draft/models/cite_wa/OneLabelTextCNN/runs/1554644075/checkpoints/model-156300"
                logger.info(checkpoint_file)
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

            current_step = sess.run(cnn.global_step)
            print("current_step: ", current_step)

    logger.info("✔︎ Done.")


if __name__ == "__main__":
    test(word2vec_path=constants.google_w2v_path)