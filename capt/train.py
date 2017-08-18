"""
参考文章：http://blog.topspeedsnail.com/archives/10858
"""
import tensorflow as tf

from capt.cfg import MAX_CAPTCHA, CHAR_SET_LEN, tb_log_path, save_model
from capt.cnn_sys import crack_captcha_cnn, Y, keep_prob, X
from capt.data_iter import get_next_batch


def train_crack_captcha_cnn():
    """
    训练模型
    :return:
    """
    output = crack_captcha_cnn()
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  # 36行，4列
    label = tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

    max_idx_p = tf.argmax(predict, 2)  # shape:batch_size,4,nb_cls
    max_idx_l = tf.argmax(label, 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)

    with tf.name_scope('my_monitor'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=label))
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    tf.summary.scalar('my_loss', loss)
    # 最后一层用来分类的softmax和sigmoid有什么不同？

    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.name_scope('my_monitor'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('my_accuracy', accuracy)

    saver = tf.train.Saver()  # 将训练过程进行保存

    sess = tf.InteractiveSession(
        config=tf.ConfigProto(
            log_device_placement=False
        )
    )

    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tb_log_path, sess.graph)

    step = 0
    while True:
        batch_x, batch_y = get_next_batch(64)  # 64
        _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.95})
        print(step, 'loss:\t', loss_)

        step += 1

        # 每2000步保存一次实验结果
        if step % 2000 == 0:
            saver.save(sess, save_model, global_step=step)

        # 在测试集上计算精度
        if step % 50 != 0:
            continue

        # 每50 step计算一次准确率，使用新生成的数据
        batch_x_test, batch_y_test = get_next_batch(256)  # 新生成的数据集个来做测试
        acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        print(step, 'acc---------------------------------\t', acc)

        # 终止条件
        if acc > 0.98:
            break

        # 启用监控 tensor board
        summary = sess.run(merged, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        writer.add_summary(summary, step)


if __name__ == '__main__':
    train_crack_captcha_cnn()
    print('end')
    pass
