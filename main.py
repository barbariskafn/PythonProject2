import tensorflow as tf

snack_tensor = tf.constant([99, 134, 245, 3, 1000000000000000000000000000], dtype=tf.float32)
weights_tensor = tf.constant([991, 4, 1, 99999999999999999999, 0.000001], dtype=tf.float32)

weighted_features = tf.multiply(snack_tensor, weights_tensor)
sum_features = tf.reduce_sum(weighted_features)

activated_features = tf.nn.relu(sum_features)
snack_rating = tf.round(activated_features)

tf.print("Рейтинг привлекательности:", snack_rating)