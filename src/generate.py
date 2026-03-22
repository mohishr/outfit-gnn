"""
generate_weights.py
───────────────────
Generates random weights for NGNN and saves them to the weights/ folder.

Usage
─────
    python generate_weights.py
"""

import os
import numpy as np
import tensorflow as tf

WEIGHT_DIR     = "weights"
CKPT_NAME      = "ngnn_best.weights.h5"
NUM_CATEGORIES = 48
EMBED_DIM      = 2048
HIDDEN_DIM     = 256


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(out_dim, use_bias=False)
        self.bn    = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x, adj = inputs
        return tf.nn.relu(self.bn(self.dense(tf.matmul(adj, x)), training=training))


class NGNNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.cat_embed  = tf.keras.layers.Embedding(NUM_CATEGORIES + 1, 64)
        self.proj       = tf.keras.layers.Dense(HIDDEN_DIM)
        self.gcn1       = GCNLayer(HIDDEN_DIM)
        self.gcn2       = GCNLayer(HIDDEN_DIM)
        self.score_head = tf.keras.layers.Dense(1)

    # Use call() so Keras tracks the build properly
    def call(self, inputs, training=False):
        embs, cat_ids, adj = inputs
        x = tf.concat([embs, self.cat_embed(cat_ids)], axis=-1)
        x = tf.nn.relu(self.proj(x))
        x = self.gcn1((x, adj), training=training)
        x = self.gcn2((x, adj), training=training)
        return tf.squeeze(self.score_head(tf.reduce_mean(x, axis=0, keepdims=True)))


# Instantiate and run one dummy forward pass through call()
# so Keras fully builds and registers all weights
model = NGNNModel()

dummy = (
    tf.zeros((4, EMBED_DIM), dtype=tf.float32),
    tf.zeros((4,),           dtype=tf.int32),
    tf.eye(4,                dtype=tf.float32),
)
_ = model(dummy, training=False)

print(f"Model built: {model.built}")
print(f"Trainable variables: {len(model.trainable_variables)}")

# Save weights
os.makedirs(WEIGHT_DIR, exist_ok=True)
ckpt_path = os.path.join(WEIGHT_DIR, CKPT_NAME)
model.save_weights(ckpt_path)

print(f"Saved random weights → {ckpt_path}")
print(f"Total params: {sum(np.prod(v.shape) for v in model.trainable_variables):,}")