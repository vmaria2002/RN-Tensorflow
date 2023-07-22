import liniar_regresion as tf

if __name__ == '__main__':
    node1 = tf.constant(4, tf.float32)
    node2 = tf.constant(6, tf.float32)

    ses = tf.Session()
    print(ses.run[node1, node2])

    ses.close()