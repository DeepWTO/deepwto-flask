# -*- coding:utf-8 -*-
__author__ = "Randolph"
__modify__ = "Zachary"

import tensorflow as tf
from utils.layers import do_cnn, fc_w_nl_bn


class OneLabelTextCNN(object):
    """A CNN for generation of text-seq encoding."""

    def __init__(
        self,
        sequence_length_gov,
        sequence_length_art,
        vocab_size,
        fc_hidden_size,
        embedding_size,
        embedding_type,
        filter_sizes,
        num_filters,
        l2_reg_lambda=0.0,
        pretrained_embedding=None,
    ):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x_gov = tf.placeholder(
            tf.int32, [None, sequence_length_gov], name="input_x_gov"
        )

        self.input_x_art = tf.placeholder(
            tf.int32, [None, sequence_length_art], name="input_x_art"
        )

        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        # Embedding Layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained
            # by our corpus
            if pretrained_embedding is None:
                self.embedding = tf.Variable(
                    tf.random_uniform(
                        [vocab_size, embedding_size],
                        minval=-1.0,
                        maxval=1.0,
                        dtype=tf.float32,
                    ),
                    trainable=True,
                    name="embedding",
                )
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(
                        pretrained_embedding, dtype=tf.float32, name="embedding"
                    )
                if embedding_type == 1:
                    self.embedding = tf.Variable(
                        pretrained_embedding,
                        trainable=True,
                        dtype=tf.float32,
                        name="embedding",
                    )
            self.embedded_sentence_gov = tf.nn.embedding_lookup(
                self.embedding, self.input_x_gov
            )

            self.embedded_sentence_art = tf.nn.embedding_lookup(
                self.embedding, self.input_x_art
            )

            self.embedded_sentence_expanded_gov = tf.expand_dims(
                self.embedded_sentence_gov, axis=-1
            )

            self.embedded_sentence_expanded_art = tf.expand_dims(
                self.embedded_sentence_art, axis=-1
            )

        # h_drop_gov_args = filter_sizes, embedding_size, num_filters

        h_drop_gov, feature_maps_gov = do_cnn(
            gov_or_art="gov",
            filter_sizes=filter_sizes,
            embedding_size=embedding_size,
            num_filters=num_filters,
            embedded_sentence_expanded=self.embedded_sentence_expanded_gov,
            is_training=self.is_training,
            sequence_length=sequence_length_gov,
            fc_hidden_size=fc_hidden_size,
            dropout_keep_prob=self.dropout_keep_prob,
        )

        print(feature_maps_gov)

        h_drop_art, feature_maps_art = do_cnn(
            gov_or_art="art",
            filter_sizes=filter_sizes,
            embedding_size=embedding_size,
            num_filters=num_filters,
            embedded_sentence_expanded=self.embedded_sentence_expanded_art,
            is_training=self.is_training,
            sequence_length=sequence_length_art,
            fc_hidden_size=fc_hidden_size,
            dropout_keep_prob=self.dropout_keep_prob,
        )

        print(tf.shape(h_drop_art))
        print(tf.shape(h_drop_gov))

        self.h_drop = tf.concat(values=[h_drop_gov, h_drop_art], axis=1)

        self.legal = fc_w_nl_bn(
            "legal",
            fc_hidden_size=2 * fc_hidden_size,
            input_tensor=self.h_drop,
            output_size=fc_hidden_size,
            is_training=self.is_training,
        )

        # Final scores
        with tf.name_scope("output"):
            W = tf.Variable(
                tf.truncated_normal(
                    shape=[fc_hidden_size, 1], stddev=0.1, dtype=tf.float32
                ),
                name="W",
            )
            b = tf.Variable(
                tf.constant(value=0.1, shape=[1], dtype=tf.float32), name="b"
            )
            self.logits = tf.nn.xw_plus_b(
                self.legal, W, b, name="logits"
            )  # this is the final logit_layer with bias
            self.scores = tf.sigmoid(self.logits, name="scores")

            def calc_grad_cam(feature_maps, seq_len):
                grad_cam_c_filtersize = []
                for feature_map in feature_maps:
                    _dy_da = tf.gradients(self.logits, feature_map)[0]
                    # shape: [None, length-filter_size+1, filter_num]

                    # squeeze after get gradients
                    _dy_da = tf.squeeze(_dy_da, [2])
                    feature_map = tf.squeeze(feature_map, [2])

                    _alpha_c = tf.reduce_mean(_dy_da, axis=1)
                    # shape: [None, filter_num]

                    _grad_cam_c = tf.nn.relu(
                        tf.reduce_sum(
                            tf.multiply(
                                tf.transpose(feature_map, perm=[0, 2, 1]),
                                tf.stack([_alpha_c], axis=2),
                            ),
                            axis=1,
                        )
                    )

                    _interpol_grad_cam_c = tf.stack(
                        [tf.stack([_grad_cam_c], axis=2)], axis=3
                    )
                    _interpol_grad_cam_c = tf.image.resize_bilinear(
                        images=_interpol_grad_cam_c,
                        size=[seq_len, 1],
                    )
                    _interpol_grad_cam_c = tf.squeeze(_interpol_grad_cam_c, axis=[2, 3])
                    # shape: [None, length]

                    grad_cam_c_filtersize.append(_interpol_grad_cam_c)

                grad_cam_c = tf.reduce_sum(tf.stack(grad_cam_c_filtersize, axis=0),
                                       axis=0)
                # grad_cam_c shape: [None, length]    (element wise sum for each grad cam per filter_size)
                grad_cam_c = grad_cam_c / tf.norm(grad_cam_c, axis=1,
                                                  keepdims=True)
                return grad_cam_c

            # grad_cam_c shape: [None, length]    (element wise normalize)
            self.grad_cam_c_gov = calc_grad_cam(feature_maps_gov, seq_len=sequence_length_gov)
            self.grad_cam_c_art = calc_grad_cam(feature_maps_art, seq_len=sequence_length_art)

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.weighted_cross_entropy_with_logits(
                targets=self.input_y, logits=self.logits, pos_weight=26.303
            )

            losses = tf.reduce_mean(
                tf.reduce_sum(losses, axis=1), name="sigmoid_losses"
            )
            l2_losses = (
                tf.add_n(
                    [
                        tf.nn.l2_loss(tf.cast(v, tf.float32))
                        for v in tf.trainable_variables()
                    ],
                    name="l2_losses",
                )
                * l2_reg_lambda
            )
            self.loss = tf.add(losses, l2_losses, name="loss")
