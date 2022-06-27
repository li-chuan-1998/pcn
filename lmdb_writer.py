'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import os
from io_util import read_pcd
from tensorpack import DataFlow, dataflow


class pcd_df(DataFlow):
    def __init__(self, target_path):
        self.root_dir = target_path

    def size(self):
        partial_path = os.path.join(self.root_dir, "partial")
        return len(os.listdir(partial_path))

    def get_data(self):
        complete_dir = os.path.join(self.root_dir, "complete")
        partial_dir = os.path.join(self.root_dir, "partial")

        for idx, file in enumerate(os.listdir(partial_dir)):
            name_com = file.split("_")
            complete_pcd = read_pcd(os.path.join(complete_dir, f"{name_com[0]}_{name_com[1]}_{name_com[2]}_complete.pcd"))
            partial = read_pcd(os.path.join(partial_dir, file))
            yield str(idx), partial, complete_pcd
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target_path')
    args = parser.parse_args()

    df = pcd_df(args.target_path)
    output_path = args.target_path + ".lmdb"
    dataflow.LMDBSerializer.save(df, output_path)
    