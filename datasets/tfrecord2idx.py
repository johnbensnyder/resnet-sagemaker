#!/usr/bin/env python
import sys
import struct
import os
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: tfrecord2idx <tfrecord filename> <index filename>")
    exit()

def create_idx(tf_record, tf_idx):
    f = open(tf_record, 'rb')
    idx = open(tf_idx, 'w')

    while True:
        current = f.tell()
        try:
            # length
            byte_len = f.read(8)
            if len(byte_len) == 0:
                break
            # crc
            f.read(4)
            proto_len = struct.unpack('q', byte_len)[0]
            # proto
            f.read(proto_len)
            # crc
            f.read(4)
            idx.write(str(current) + ' ' + str(f.tell() - current) + '\n')
        except Exception:
            print("Not a valid TFRecord file")
            break

    f.close()
    idx.close()

if __name__=='__main__':
    tf_record_dir = Path(sys.argv[1])
    tf_idx_dir = Path(sys.argv[2])
    tf_idx_dir.mkdir()
    for tf_record in tf_record_dir.glob('*'):
        create_idx(tf_record, tf_idx_dir.joinpath('{}.idx'.format(tf_record.stem)))
