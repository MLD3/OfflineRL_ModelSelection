import tensorflow as tf

def select_output(args):
    out, a = args
    return tf.gather(out, a, axis=1, batch_dims=1)

def select_output_d(args, d):
    out, a = args
    inds = tf.map_fn(fn=lambda a_i: tf.range(d*a_i, d*(a_i+1)), elems=a)
    return tf.gather(out, inds, axis=1, batch_dims=1)
