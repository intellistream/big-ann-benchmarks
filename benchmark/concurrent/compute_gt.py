import argparse
import os
import numpy as np
import PyCANDYAlgo

import sys
[sys.path.append(i) for i in ['.', '..']]

from benchmark.datasets import DATASETS
from benchmark.concurrent.load_runbook import load_runbook_concurrent

        
def gt_dir(ds, runbook_path):
    runbook_filename = os.path.split(runbook_path)[1]
    return os.path.join(ds.basedir, str(ds.nb), runbook_filename)


def get_next_set(id_list: list, entry):
    match entry['operation']:
        case 'initial' | 'insert_and_search':
            id_list.extend(range(entry['start'], entry['end']))
            return id_list
        case 'search':
            return id_list
        case _:
            raise ValueError('Undefined entry in runbook')


def output_gt(ds, id_list, step, gt_cmdline, runbook_path):
    ids = np.array(id_list, dtype=np.uint32)

    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[ids]  

    dir = gt_dir(ds, runbook_path)
    prefix = os.path.join(dir, f'step{step}')
    os.makedirs(dir, exist_ok=True)

    data_file = prefix + '.data'
    gt_file = prefix + '.gt100'

    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little'))  
        f.write(ds.d.to_bytes(4, byteorder='little'))  
        data_slice.tofile(f)

    gt_cmd = f"{gt_cmdline} --base_file {data_file} --gt_file {gt_file}"
    print(f"Executing GT command: {gt_cmd}")
    os.system(gt_cmd)
    
    print(gt_file)

    print(f"Removing data file: {data_file}")
    os.system(f"rm {data_file}")


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
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    max_pts, cc_config, runbook  = load_runbook_concurrent(args.dataset, ds.nb, args.runbook_file)
    query_file = ds.qs_fn if args.private_query else ds.qs_fn
    
    # compute final search recall
    common_cmd = args.gt_cmdline_tool + ' --dist_fn ' 
    match ds.distance():
        case 'euclidean': common_cmd += 'l2'
        case 'ip': common_cmd += 'mips'
        case _: raise RuntimeError('Invalid metric')
    common_cmd += ' --data_type '
    match ds.dtype:
        case 'float32': common_cmd += 'float'
        case 'int8': common_cmd += 'int8'
        case 'uint8': common_cmd += 'uint8'
        case _: raise RuntimeError('Invalid datatype')
    common_cmd += ' --K 100'
    common_cmd += ' --query_file ' + os.path.join(ds.basedir, query_file)
    
    step = 1
    id_list = []
    
    for entry in runbook:
        id_list = get_next_set(id_list, entry)
        
        if entry['operation'] == 'search':
            output_gt(ds, id_list, step, common_cmd, args.runbook_file)
            
        # elif entry['operation'] == '':
        #     output_stepwise_gt(ds, tag_to_id, step, common_cmd, args.runbook_file)

        step += 1


if __name__ == '__main__':
    main()