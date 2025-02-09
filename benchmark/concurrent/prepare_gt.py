import argparse
import os
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]

from benchmark.datasets import DATASETS
from benchmark.concurrent.load_runbook import load_runbook_concurrent

        
def gt_dir(ds, runbook_path):
    runbook_filename = os.path.split(runbook_path)[1]
    return os.path.join(ds.basedir, str(ds.nb), runbook_filename)


def output_gt(ds, tag_to_id, step, gt_cmdline, runbook_path):
    ids = np.array(list(tag_to_id.values()), dtype=np.uint32)
    tags = np.array(list(tag_to_id.keys()), dtype=np.uint32)

    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[ids]  

    dir = gt_dir(ds, runbook_path)
    prefix = os.path.join(dir, f'step{step}')
    os.makedirs(dir, exist_ok=True)

    tags_file = prefix + '.tags'
    data_file = prefix + '.data'
    gt_file = prefix + '.gt100'

    with open(tags_file, 'wb') as tf:
        tf.write(tags.size.to_bytes(4, byteorder='little'))
        tf.write((1).to_bytes(4, byteorder='little'))
        tags.tofile(tf)

    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little'))  
        f.write(ds.d.to_bytes(4, byteorder='little'))  
        data_slice.tofile(f)

    gt_cmd = f"{gt_cmdline} --base_file {data_file} --gt_file {gt_file} --tags_file {tags_file}"
    print(f"Executing GT command: {gt_cmd}")
    os.system(gt_cmd)

    print(f"Removing data file: {data_file}")
    os.system(f"rm {data_file}")


def output_gt_batch(ds, tag_to_id, num_batch_insert, step, gt_cmdline, runbook_path, batchSize, with_deletion = False):
    ids_list = []
    tags_list = []
    for tag, id in tag_to_id.items():
        ids_list.append(id)
        tags_list.append(tag)

    ids = np.array(ids_list, dtype=np.uint32)
    tags = np.array(tags_list, dtype=np.uint32)

    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[np.array(ids)]

    dir = gt_dir(ds, runbook_path)
    prefix = os.path.join(dir, 'batch') + str(num_batch_insert)+"_"+str(step)
    os.makedirs(dir, exist_ok=True)

    tags_file = prefix + '.tags'
    data_file = prefix + '.data'
    gt_file = prefix + '.gt100'

    with open(tags_file, 'wb') as tf:
        one = 1
        tf.write(tags.size.to_bytes(4, byteorder='little'))
        tf.write(one.to_bytes(4, byteorder='little'))
        tags.tofile(tf)
    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little'))  # npts
        f.write(ds.d.to_bytes(4, byteorder='little'))
        data_slice.tofile(f)

    gt_cmdline += ' --base_file ' + data_file
    gt_cmdline += ' --gt_file ' + gt_file
    gt_cmdline += ' --tags_file ' + tags_file
    print("Executing cmdline: ", gt_cmdline)
    os.system(gt_cmdline)
    print("Removing data file")
    rm_cmdline = "rm " + data_file
    os.system(rm_cmdline)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        help=f'Dataset to benchmark on.',
        required=True)
    parser.add_argument(
        '--runbook_file',
        help='Runbook yaml file path'
    )
    parser.add_argument(
        '--private_query',
        action='store_true'
    )
    parser.add_argument(
        '--gt_cmdline_tool',
        required=True
    )
    parser.add_argument(
        '--download',
        action='store_true'
    )
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    _, batch_size, url, runbook  = load_runbook_concurrent(args.dataset, ds.nb, args.runbook_file)
    query_file = ds.qs_fn if args.private_query else ds.qs_fn
    
    common_cmd = args.gt_cmdline_tool + ' --dist_fn ' 
    match ds.distance():
        case 'euclidean':
            common_cmd += 'l2'
        case 'ip':
            common_cmd += 'mips'
        case _:
            raise RuntimeError('Invalid metric')
    common_cmd += ' --data_type '
    match ds.dtype:
        case 'float32':
            common_cmd += 'float'
        case 'int8':
            common_cmd += 'int8'
        case 'uint8':
            common_cmd += 'uint8'
        case _:
            raise RuntimeError('Invalid datatype')
    common_cmd += ' --K 100'
    common_cmd += ' --query_file ' + os.path.join(ds.basedir, query_file)

    step = 1
    tag_to_id = {}
    insert_count = 0
    
    for entry in runbook:
        if entry['operation'] != 'insert_and_search':
            raise ValueError(f"Unsupported operation: {entry['operation']}")
        
        start, end = entry['start'], entry['end']
        for batch_start in range(start, end, batch_size):
            batch_end = min(batch_start + batch_size, end)
            
            for i in range(batch_start, batch_end):
                tag_to_id[i] = i
                insert_count += 1
                
            print(f"Inserted {batch_size} items (Total: {insert_count})")
            print(f"Computing GT at {insert_count} insertions")
            
            output_gt(ds, tag_to_id, step, common_cmd, args.runbook_file)

        step += 1

if __name__ == '__main__':
    main()