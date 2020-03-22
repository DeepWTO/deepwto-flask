# -*- coding:utf-8 -*-
__author__ = "Randolph"
__modify__ = "Zachary"

import sys
import time
import logging
import tensorflow as tf

from model import OneLabelTextCNN
import feed
import utils
import constants

TRAIN_OR_RESTORE = "R"

logging.info("✔︎ The format of your input is legal, " "now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == "T":
    logger = utils.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()))
if TRAIN_OR_RESTORE == "R":
    logger = utils.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()))

VALIDATIONSET_DIR = "/home/zachary/139_[Article III:4].pkl"

# Data Parameters

tf.flags.DEFINE_string(
    "validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data."
)

tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")

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

tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training Parameters
tf.flags.DEFINE_integer(
    "batch_size",
    1,
    # 8 for zacbuntu
    "Batch Size (default: 256)",
)

tf.flags.DEFINE_string(
    "checkpoint_model_path",
    constants.ckpt_model_path,
    "Batch Size (default: 256)",
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


def test(word2vec_path, data):
    """Training TextCNN-one-label model."""

    # Load sentences, labels, and training parameters
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Validation data processing...")
    val_data = feed.load_data_and_labels_one_label(
        data,
        word2vec_path=word2vec_path,
        use_pretrain=True
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
    print("x_val_testid", len(x_val_testid), x_val_testid[0])
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

            saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=FLAGS.num_checkpoints
            )  # this is required to resolve "tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value Global_Step [[{{node _retval_Global_Step_0_0}}]]"

            if FLAGS.train_or_restore == "R":
                # Load cnn model
                logger.info("✔︎ Loading model...")
                # checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                # logger.info("model_path", FLAGS.checkpoint_model_path)
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(FLAGS.checkpoint_model_path))
                print(FLAGS.checkpoint_model_path)
                saver.restore(sess, FLAGS.checkpoint_model_path)

            current_step = sess.run(cnn.global_step)
            print("current_step: ", current_step)

            def infer(
                _x_val_testid, _x_val_gov, _x_val_art, _y_val, writer=None
            ):
                print("_x_val_gov: ", len(_x_val_gov), _x_val_gov)

                print("_x_val_art: ", len(_x_val_art))
                """Evaluates model on a validation set"""
                batches_validation = feed.batch_iter(
                    list(zip(_x_val_testid, _x_val_gov, _x_val_art, _y_val)),
                    FLAGS.batch_size,
                    num_epochs=1,
                    shuffle=False,
                )

                valid_step_count = 0
                for batch_validation in batches_validation:
                    print(valid_step_count)

                    x_val_testid, x_batch_val_gov, x_batch_val_art, y_batch_val = zip(
                        *batch_validation
                    )

                    feed_dict = {
                        cnn.input_x_gov: x_batch_val_gov,
                        cnn.input_x_art: x_batch_val_art,
                        cnn.input_y: y_batch_val,
                        cnn.dropout_keep_prob: 1.0,
                        cnn.is_training: False,
                    }

                    # print(feed_dict)

                    [
                    step,
                    scores,
                    grad_cam_c_gov,
                    grad_cam_c_art,
                    cur_loss,
                    input_y
                    ] = sess.run(
                        [
                            cnn.global_step,
                            cnn.scores,
                            cnn.grad_cam_c_gov,
                            cnn.grad_cam_c_art,
                            cnn.loss,
                            cnn.input_y
                        ],
                        feed_dict,
                    )

                    print(x_val_testid, scores, grad_cam_c_gov, grad_cam_c_art)

            infer(
                x_val_testid,
                x_val_gov,
                x_val_art,
                y_val,
                writer=None,
            )


if __name__ == "__main__":
    test(word2vec_path=constants.google_w2v_path,
         data="/Users/zachary/Downloads/test_data.json")
