import argparse
import importlib
import os
import tensorflow as tf

from io_util import read_pcd, save_pcd
from data_util import resample_pcd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, help='path to the input directory')
    parser.add_argument('-o', '--output_path', type=str, help='path to the output directory')
    parser.add_argument('-m', '--model_type', type=str, default='pcn_emd', help='model type')
    parser.add_argument('-c', '--checkpoint', type=str, default='training/pcn_emd')
    parser.add_argument('-n', '--num_input_pts', type=int, default=8192)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    output_pts = 16384
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    gt = tf.placeholder(tf.float32, (1, output_pts, 3))
    npts = tf.placeholder(tf.int32, (1,))
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    os.makedirs(args.output_path, exist_ok=True)
    
    for file in os.listdir(args.input_folder):
        partial = resample_pcd(read_pcd(os.path.join(args.input_folder, file)), args.num_input_pts)
        complete = sess.run(model.outputs, feed_dict={inputs: [partial], npts: [partial.shape[0]]})[0]

        output_file = os.path.join(args.output_path, file.replace("partial",'output'))
        save_pcd(output_file, complete)


if __name__ == '__main__':
    main()
