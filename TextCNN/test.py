# -*- coding:utf-8 -*-
__author__ = "Randolph"
__modify__ = "Zachary"

import os
import sys
import time
import pickle
import logging
import numpy as np
import tensorflow as tf

from tensorboard.plugins import projector
from model import OneLabelTextCNN
from utils import feed
from utils.train import count_correct_pred

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import matplotlib.pyplot as plt

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
    logger = feed.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()))
if TRAIN_OR_RESTORE == "R":
    logger = feed.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()))

VALIDATIONSET_DIR = "/home/zachary/tmp/test_data.json"
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
        word2vec_path=word2vec_path,
        data_aug_flag=False,
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

    VOCAB_SIZE = feed.load_vocab_size(FLAGS.embedding_dim, word2vec_path=word2vec_path)

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
                train_op = optimizer.apply_gradients(
                    zip(grads, variables), global_step=cnn.global_step, name="train_op"
                )

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

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", cnn.loss)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(
                validation_summary_dir, sess.graph
            )

            saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=FLAGS.num_checkpoints
            )

            if FLAGS.train_or_restore == "R":
                # Load cnn model
                logger.info("✔︎ Loading model...")
                print(checkpoint_dir)
                # checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                checkpoint_file = "/home/zachary/tmp/1554644075/checkpoints/model-156300"
                logger.info(checkpoint_file)
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Embedding visualization config
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = "embedding"
                embedding_conf.metadata_path = FLAGS.metadata_file

                projector.visualize_embeddings(validation_summary_writer, config)

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(cnn.global_step)
            print("current_step: ", current_step)

            def validation_step(
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

                _eval_counter, _eval_loss = 0, 0.0

                _eval_pre_tk = [0.0] * FLAGS.top_num
                _eval_rec_tk = [0.0] * FLAGS.top_num
                _eval_F_tk = [0.0] * FLAGS.top_num

                true_onehot_labels = []
                predicted_onehot_scores = []
                predicted_onehot_labels_ts = []
                predicted_onehot_labels_tk = [[] for _ in range(FLAGS.top_num)]

                valid_count_correct_one = 0
                valid_count_label_one = 0
                valid_count_correct_zero = 0
                valid_count_label_zero = 0

                valid_step_count = 0
                for batch_validation in batches_validation:
                    print(valid_step_count)

                    x_val_testid, x_batch_val_gov, x_batch_val_art, y_batch_val = zip(
                        *batch_validation
                    )

                    art = x_val_testid[0].split("_")[-1].split(" ")[-1]

                    if len(art) >= 3:
                        if art[0:3] == "III":
                            if y_batch_val[0][0] > 0:

                                feed_dict = {
                                    cnn.input_x_gov: x_batch_val_gov,
                                    cnn.input_x_art: x_batch_val_art,
                                    cnn.input_y: y_batch_val,
                                    cnn.dropout_keep_prob: 1.0,
                                    cnn.is_training: False,
                                }
                                (
                                    step,
                                    summaries,
                                    scores,
                                    grad_cam_c_gov,
                                    grad_cam_c_art,
                                    cur_loss,
                                    input_y,
                                ) = sess.run(
                                    [
                                        cnn.global_step,
                                        validation_summary_op,
                                        cnn.scores,
                                        cnn.grad_cam_c_gov,
                                        cnn.grad_cam_c_art,
                                        cnn.loss,
                                        cnn.input_y,
                                    ],
                                    feed_dict,
                                )

                                (
                                    count_label_one,
                                    count_label_zero,
                                    count_correct_one,
                                    count_correct_zero,
                                    TFPN,
                                ) = count_correct_pred(scores, input_y)

                                def _plot_score(
                                    vec, pred_text, xticks, gov_or_art, testid
                                ):
                                    _axis_fontsize = 13
                                    fig = plt.figure(figsize=(14, 10))
                                    plt.yticks([])
                                    plt.xticks(
                                        range(0, len(vec)),
                                        xticks,
                                        fontsize=_axis_fontsize,
                                    )
                                    fig.add_subplot(1, 1, 1)
                                    plt.figtext(
                                        x=0.13,
                                        y=0.54,
                                        s="Prediction: {}".format(pred_text),
                                        fontsize=15,
                                        fontname="sans-serif",
                                    )
                                    img = plt.imshow([vec], vmin=0, vmax=1)
                                    # plt.show()
                                    plt.savefig(testid[0] + "_" + gov_or_art + ".png")

                                raw_gov_tokens = val_data.raw_tokens_gov[
                                    valid_step_count
                                ]
                                raw_art_tokens = val_data.raw_tokens_art[
                                    valid_step_count
                                ]

                                print(x_val_testid)
                                print(grad_cam_c_gov[0], len(grad_cam_c_gov[0]))
                                print(grad_cam_c_art[0], len(grad_cam_c_art[0]))
                                print(raw_gov_tokens)
                                print(raw_art_tokens)

                                if TFPN == "TRUE POSITIVE":
                                    pkl_target = dict()
                                    pkl_target["x_val_testid"] = x_val_testid
                                    pkl_target["grad_cam_c_gov"] = grad_cam_c_gov[0]
                                    pkl_target["grad_cam_c_art"] = grad_cam_c_art[0]
                                    pkl_target["raw_gov_tokens"] = raw_gov_tokens
                                    pkl_target["raw_art_tokens"] = raw_art_tokens
                                    with open(
                                        x_val_testid[0] + "_grad_cams" + ".pkl",
                                        "wb",
                                    ) as handle:
                                        pickle.dump(
                                            pkl_target, handle, protocol=pickle.HIGHEST_PROTOCOL
                                        )

                                # _plot_score(grad_cam_c[0], pred_text="POSITVE", xticks=raw_gov_tokens, gov_or_art="gov", testid=x_val_testid)
                                # _plot_score(grad_cam_c[0], pred_text="POSITVE", xticks=raw_art_tokens, gov_or_art="art", testid=x_val_testid)

                                valid_count_correct_one += count_correct_one
                                valid_count_label_one += count_label_one

                                valid_count_correct_zero += count_correct_zero
                                valid_count_label_zero += count_label_zero

                                print(
                                    "[VALID] num_correct_answer is {} out of {}".format(
                                        count_correct_one, count_label_one
                                    )
                                )
                                print(
                                    "[VALID] num_correct_answer is {} out of {}".format(
                                        count_correct_zero, count_label_zero
                                    )
                                )

                                # Prepare for calculating metrics
                                for i in y_batch_val:
                                    true_onehot_labels.append(i)
                                for j in scores:
                                    predicted_onehot_scores.append(j)

                                # Predict by threshold
                                batch_predicted_onehot_labels_ts = feed.get_onehot_label_threshold(
                                    scores=scores, threshold=FLAGS.threshold
                                )

                                for k in batch_predicted_onehot_labels_ts:
                                    predicted_onehot_labels_ts.append(k)

                                # Predict by topK
                                for _top_num in range(FLAGS.top_num):
                                    batch_predicted_onehot_labels_tk = feed.get_onehot_label_topk(
                                        scores=scores, top_num=_top_num + 1
                                    )

                                    for i in batch_predicted_onehot_labels_tk:
                                        predicted_onehot_labels_tk[_top_num].append(i)

                                _eval_loss = _eval_loss + cur_loss
                                _eval_counter = _eval_counter + 1

                                if writer:
                                    writer.add_summary(summaries, step)
                            else:
                                pass

                    valid_step_count += 1

                logger.info(
                    "[VALID_FINAL] Total Correct One Answer is {} out "
                    "of {}".format(valid_count_correct_one, valid_count_label_one)
                )
                logger.info(
                    "[VALID_FINAL] Total Correct Zero Answer is {} "
                    "out of {}".format(valid_count_correct_zero, valid_count_label_zero)
                )

                _eval_loss = float(_eval_loss / _eval_counter)

                # Calculate Precision & Recall & F1 (threshold & topK)
                _eval_pre_ts = precision_score(
                    y_true=np.array(true_onehot_labels),
                    y_pred=np.array(predicted_onehot_labels_ts),
                    average="micro",
                )
                _eval_rec_ts = recall_score(
                    y_true=np.array(true_onehot_labels),
                    y_pred=np.array(predicted_onehot_labels_ts),
                    average="micro",
                )
                _eval_F_ts = f1_score(
                    y_true=np.array(true_onehot_labels),
                    y_pred=np.array(predicted_onehot_labels_ts),
                    average="micro",
                )

                for _top_num in range(FLAGS.top_num):
                    _eval_pre_tk[_top_num] = precision_score(
                        y_true=np.array(true_onehot_labels),
                        y_pred=np.array(predicted_onehot_labels_tk[_top_num]),
                        average="micro",
                    )
                    _eval_rec_tk[_top_num] = recall_score(
                        y_true=np.array(true_onehot_labels),
                        y_pred=np.array(predicted_onehot_labels_tk[_top_num]),
                        average="micro",
                    )
                    _eval_F_tk[_top_num] = f1_score(
                        y_true=np.array(true_onehot_labels),
                        y_pred=np.array(predicted_onehot_labels_tk[_top_num]),
                        average="micro",
                    )

                # Calculate the average AUC
                _eval_auc = roc_auc_score(
                    y_true=np.array(true_onehot_labels),
                    y_score=np.array(predicted_onehot_scores),
                    average="micro",
                )
                # Calculate the average PR
                _eval_prc = average_precision_score(
                    y_true=np.array(true_onehot_labels),
                    y_score=np.array(predicted_onehot_scores),
                    average="micro",
                )

                return (
                    _eval_loss,
                    _eval_auc,
                    _eval_prc,
                    _eval_rec_ts,
                    _eval_pre_ts,
                    _eval_F_ts,
                    _eval_rec_tk,
                    _eval_pre_tk,
                    _eval_F_tk,
                )

            logger.info("\nEvaluation:")
            (
                eval_loss,
                eval_auc,
                eval_prc,
                eval_rec_ts,
                eval_pre_ts,
                eval_F_ts,
                eval_rec_tk,
                eval_pre_tk,
                eval_F_tk,
            ) = validation_step(
                x_val_testid,
                x_val_gov,
                x_val_art,
                y_val,
                writer=validation_summary_writer,
            )

            logger.info(
                "All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}".format(
                    eval_loss, eval_auc, eval_prc
                )
            )

            # Predict by threshold
            logger.info(
                "☛ Predict by threshold: Precision {0:g}, Recall {1:g}, "
                "F {2:g}".format(eval_pre_ts, eval_rec_ts, eval_F_ts)
            )

            # Predict by topK
            logger.info("☛ Predict by topK:")
            for top_num in range(FLAGS.top_num):
                logger.info(
                    "Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}".format(
                        top_num + 1,
                        eval_pre_tk[top_num],
                        eval_rec_tk[top_num],
                        eval_F_tk[top_num],
                    )
                )
            # best_saver.handle(eval_prc, sess, current_step)

    logger.info("✔︎ Done.")


if __name__ == "__main__":
    test(word2vec_path="/home/zachary/tmp/GoogleNews-vectors-negative300.bin")
