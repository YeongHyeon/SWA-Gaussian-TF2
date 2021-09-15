import argparse, time, os, operator

import tensorflow as tf
import source.utils as utils
import source.connector as con
import source.tf_process as tfp
import source.datamanager as dman

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    dataset = dman.DataSet()

    agent = con.connect(nn=FLAGS.nn).Agent(\
        dim_h=dataset.dim_h, dim_w=dataset.dim_w, dim_c=dataset.dim_c, num_class=dataset.num_class, \
        ksize=FLAGS.ksize, filters=FLAGS.filters, learning_rate=FLAGS.lr, k_max=FLAGS.k_max, \
        path_ckpt='Checkpoint')

    time_tr = time.time()
    tfp.training(agent=agent, dataset=dataset, \
        batch_size=FLAGS.batch, epochs=FLAGS.epochs, c_update=FLAGS.c_update, s_avg=FLAGS.s_avg)
    time_fin = time.time()
    tr_time = time_fin - time_tr

    print("Time (TR): %.5f [sec]" %(tr_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0", help='')

    parser.add_argument('--nn', type=int, default=0, help='')

    parser.add_argument('--ksize', type=int, default=3, help='')
    parser.add_argument('--filters', type=str, default="16,32,64", help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')

    parser.add_argument('--batch', type=int, default=32, help='')
    parser.add_argument('--epochs', type=int, default=300, help='')

    parser.add_argument('--k_max', type=int, default=10, help='maximum number of columns in deviation matrix')
    parser.add_argument('--c_update', type=int, default=10, help='moment update frequency')
    parser.add_argument('--s_avg', type=int, default=30, help='number of samples in Bayesian model averaging')

    FLAGS, unparsed = parser.parse_known_args()

    main()
