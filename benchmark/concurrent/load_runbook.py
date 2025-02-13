import os
import yaml
from collections import namedtuple

def load_runbook_concurrent(dataset_name, max_pts, runbook_file):
    with open(runbook_file) as fd:
        runbook = yaml.safe_load(fd)[dataset_name] 

    run_list = []
    i = 1
    while str(i) in runbook:  
        entry = runbook[str(i)]
        
        if entry['operation'] not in {'insert_and_search'}:
            raise Exception('Undefined runbook operation')

        if 'start' not in entry or 'end' not in entry:
            raise Exception(f'Start/End missing in runbook at entry {i}')
        
        if entry['start'] < 0 or entry['start'] >= max_pts:
            raise Exception(f'Start out of range at entry {i}')
        if entry['end'] < 0 or entry['end'] > max_pts:
            raise Exception(f'End out of range at entry {i}')

        run_list.append(entry)
        i += 1
        
    max_pts = runbook.get('max_pts')
    if max_pts == None:
        raise Exception('max points not listed for dataset in runbook')
    
    write_ratio = runbook.get('write_ratio')
    if write_ratio == None:
        raise Exception('write threads ratio not listed in runbook')

    batch_size = runbook.get('batch_size')
    if batch_size == None:
        raise Exception('batch size not listed in runbook')

    num_threads = runbook.get('num_threads')
    if num_threads == None:
        num_threads = os.cpu_count()
        print(f"number of threads not listed in runbook, use default threads {num_threads}")
    
    cc_config = {
        'write_ratio': write_ratio,
        'batch_size': batch_size,
        'num_threads': num_threads
    }

    return max_pts, cc_config, run_list

def get_gt_url(dataset_name, runbook_file):
    with open(runbook_file) as fd:
        runbook = yaml.safe_load(fd)[dataset_name]
        return runbook.get('gt_url', "none")  
