import numpy as np
import time
import os

from benchmark.algorithms.base_runner import BaseRunner
from benchmark.datasets import DATASETS
    
class ConcurrentRunner(BaseRunner):
    def build(algo, dataset, max_pts, cc_config):
        t0 = time.time()
        ds = DATASETS[dataset]()
        ndims = ds.d
        algo.setup(ds.dtype, max_pts, cc_config, ndims)
        print('Algorithm set up')
        return time.time() - t0
        
    def run_task(algo, ds, distance, count, run_count, search_type, private_query, runbook, definition, query_arguments, runbook_path, dataset):
        all_cc_results = []
        all_results = []
        cc_time = 0

        Q = ds.get_queries() if not private_query else ds.get_private_queries()
        print(fr"Got {Q.shape[0]} queries")  

        print(runbook)
        for step, entry in enumerate(runbook):
            start_time = time.time()
            match entry['operation']:
                case 'initial':
                    start = entry['start']
                    end = entry['end']
                    algo.initial(ds.get_data_in_range(start, end))
                case 'insert_and_search':
                    start = entry['start']
                    end = entry['end']
                    algo.cc_insert_and_query(ds.get_data_in_range(start, end), Q, count)
                    cc_time += (time.time() - start_time)
                case 'search':
                    algo.query(Q, count)
                    all_results.append(results = algo.get_results())
                case _:
                    raise NotImplementedError('Invalid runbook operation.')
            step_time = (time.time() - start_time)
            print(f"Step {step+1} took {step_time}s.")
            
        cc_res_file = algo.get_cc_results()
        
        cc_config = algo.get_cc_config()
        
        attrs = {
            "name": str(algo),
            "run_count": run_count,
            "distance": distance,
            "type": search_type,
            "count": int(count),
            "private_queries": private_query,
            "cc_time": cc_time, 
            "num_threads": cc_config["num_threads"],
            "write_batch": cc_config["batch_size"],
            "write_ratio": cc_config["write_ratio"],
            "cc_result_filemae": cc_res_file,
        }
            
        return (attrs, all_results)
    
    
    
