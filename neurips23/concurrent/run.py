import numpy as np
import time
import os

from benchmark.algorithms.base_runner import BaseRunner
from benchmark.datasets import DATASETS
    
class ConcurrentRunner(BaseRunner):
    def build(algo, dataset, max_pts, cc_config):
        '''
        Return set up time
        '''
        t0 = time.time()
        ds = DATASETS[dataset]()
        ndims = ds.d
        algo.setup(ds.dtype, max_pts, cc_config, ndims)
        print('Algorithm set up')
        return time.time() - t0
        
    def run_task(algo, ds, distance, count, run_count, search_type, private_query, runbook, definition, query_arguments, runbook_path, dataset):
        all_cc_results = []
        all_results = []
        step_times = []

        Q = ds.get_queries() if not private_query else ds.get_private_queries()
        print(fr"Got {Q.shape[0]} queries")  

        for step, entry in enumerate(runbook):
            start_time = time.time()
            match entry['operation']:
                case 'insert_and_search':
                    start = entry['start']
                    end = entry['end']
                    algo.cc_insert_and_query(ds.get_data_in_range(start, end), Q, count)
                    all_cc_results.append(algo.get_cc_results())
                case 'search':
                    algo.query(Q, count)
                    all_results.append(results = algo.get_results())
                case _:
                    raise NotImplementedError('Invalid runbook operation.')
            step_time = (time.time() - start_time)
            step_times[step] = step_time
            print(f"Step {step+1} took {step_time}s.")

        attrs = {
            "name": str(algo),
            "type": search_type,
        }
            
        return (attrs, all_cc_results, all_results)
    
    
    
