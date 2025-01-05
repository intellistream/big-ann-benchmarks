import numpy as np
import time
import yaml

from benchmark.algorithms.base_runner import BaseRunner
from benchmark.datasets import DATASETS

from concurrent.futures import ThreadPoolExecutor, as_completed

class ConcurrentRunner(BaseRunner):
    def build(algo, dataset, max_pts):
        '''
        Return set up time
        '''
        t0 = time.time()
        ds = DATASETS[dataset]()
        ndims = ds.d
        algo.setup(ds.dtype, max_pts, ndims)
        print('Algorithm set up')
        return time.time() - t0
    
        
    def run_task(algo, ds, distance, count, run_count, search_type, private_query, runbook):
        best_search_time = float('inf')
        search_times = []
        all_results = []

        Q = ds.get_queries() if not private_query else ds.get_private_queries()
        print(fr"Got {Q.shape[0]} queries")  

        # Load Runbook
        result_map = {}
        num_searches = 0
        
        with ThreadPoolExecutor() as executor:
            futures = []
            
            for step, entry in enumerate(runbook):
                start_time = time.time()
            
                match entry['operation']:
                    case 'insert_and_search':
                        start = entry['start']
                        end = entry['end']
                        
                        futures.append(executor.submit(algo.insert_data, algo, ds, start, end))
                        futures.append(executor.submit(algo.search_data, algo, Q, count, result_map, num_searches))
                        
                    case _:
                        raise NotImplementedError(f"Operation '{entry['operation']}' not supported.")
                
                step_time = (time.time() - start_time)
                print(f"Step {step+1} took {step_time}s.")
            
            for future in as_completed(futures):
                all_results.append(future.result())

        attrs = {
            "name": str(algo),
            "run_count": run_count,
            "distance": distance,
            "type": search_type,
            "count": int(count),
            "search_times": search_times,
            "num_searches": num_searches,
            "private_queries": private_query, 
        }

        for k, v in result_map.items():
            attrs['step_' + str(k)] = v

        additional = algo.get_additional()
        for k in additional:
            attrs[k] = additional[k]
        return (attrs, all_results)