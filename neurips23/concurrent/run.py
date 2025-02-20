import numpy as np
import time
import os

from benchmark.algorithms.base_runner import BaseRunner
from benchmark.datasets import DATASETS
from benchmark.results import get_cc_result_filename
    
class ConcurrentRunner(BaseRunner):
    def build(algo, dataset, max_pts, cc_config):
        t0 = time.time()
        ds = DATASETS[dataset]()
        ndims = ds.d
        algo.setup(ds.dtype, max_pts, cc_config, ndims)
        print('Algorithm set up')
        return time.time() - t0
        
    def run_task(algo, ds, distance, count, run_count, search_type, private_query, runbook, definition, query_arguments, runbook_path, dataset):
        all_results = []
        cc_time = 0

        Q = ds.get_queries() if not private_query else ds.get_private_queries()
        print(fr"Got {Q.shape[0]} queries")  
        
        cc_config = algo.get_cc_config()
        cc_res_file = get_cc_result_filename(dataset, count, definition, query_arguments, neurips23track="concurrent", runbook_path=runbook_path, cc_config=cc_config)

        result_map = {}
        num_searches = 0
        for step, entry in enumerate(runbook):
            start_time = time.time()
            match entry['operation']:
                case 'initial':
                    print("initial")
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
                    all_results.append(algo.get_results())
                    result_map[num_searches] = step + 1
                    num_searches += 1
                case _:
                    raise NotImplementedError('Invalid runbook operation.')
            step_time = (time.time() - start_time)
            print(f"Step {step+1} took {step_time}s.")
            
        algo.save_cc_results(cc_res_file)
        
        attrs = {
            "name": str(algo),
            "run_count": run_count,
            "distance": distance,
            "type": search_type,
            "count": int(count),
            "private_queries": private_query,
            "num_searches": num_searches,
            "cc_time": cc_time, 
            "batch_size": cc_config["batch_size"],
            "write_ratio": cc_config["write_ratio"],
            "num_threads": cc_config["num_threads"],
            "cc_result_filemae": cc_res_file,
        }
        
        for k, v in result_map.items():
            attrs['step_' + str(k)] = v

        additional = algo.get_additional()
        for k in additional:
            attrs[k] = additional[k]
            
        return (attrs, all_results)
    
    
    
