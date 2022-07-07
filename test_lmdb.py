from data_util import lmdb_dataflow

train_path = "/home/wrs/pcn/dataset_v2/train.lmdb"
valid_path = "/home/wrs/pcn/dataset_v2/valid.lmdb"
bs = 8
num_inputs = 8192
num_gt = 16384


if __name__ == '__main__':
    df_train, num_train = lmdb_dataflow(
        train_path, 8, num_inputs, num_gt, is_training=True)
    train_gen = df_train.get_data()
    df_valid, num_valid = lmdb_dataflow(
        valid_path, 8, num_inputs, num_gt, is_training=False)
    valid_gen = df_valid.get_data()

    ids, inputs, npts, gt = next(train_gen)
    print("Main: ",len(ids), len(inputs), len(npts), len(gt))
    print([len(x) for x in inputs])