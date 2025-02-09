import numpy as np
import time
import yaml
import queue
import concurrent.futures

from benchmark.algorithms.base_runner import BaseRunner
from benchmark.datasets import DATASETS

from benchmark.results import get_result_filename
from concurrent.futures import ThreadPoolExecutor, as_completed


def generateTimestamps(rows, eventRate=4000):
    """
    generates uniformly increasing event timestamps and processing timestamps for each row of the input batch vectors
    :param rows: int -
    :param eventRate: float
    :return: tuple - (eventTimestamps, processingTimestamps)
    """
    # Calculate time gap in ms
    staticDataSet = False
    intervalMicros = int(1e6 / eventRate)

    numRows = rows
    eventTimestamps = None
    if (staticDataSet):
        # generate processing timestampes and initialize as all 0s
        eventTimestamps = np.zeros(numRows, dtype=int)
    else:
        # generate uniformly increasing event arrival times
        eventTimestamps = np.arange(0, numRows * intervalMicros, intervalMicros, dtype=int)
    return eventTimestamps


def getLatencyPercentile(fraction: float, event_time: np.ndarray, processed_time: np.ndarray):
    """
    Calculate the latency percentile from event and processed time tensors.
    :param fraction: float - Percentile in the range 0 ~ 1
    :param event_time: torch.Tensor - int64 tensor of event arrival timestamps
    :param processed_time: torch.Tensor - int64 tensor of processed timestamps
    :return: int - The latency value at the specified percentile

    valid_latency = (processed_time - event_time)[(processed_time >= event_time) & (processed_time != 0)]
    """

    # If no valid latency, return 0 as in the C++ code
    if valid_latency.size == 0:
        print("No valid latency found")
        valid_latency = 0

    # Sort the valid latency values
    valid_latency_sorted = np.sort(valid_latency)

    # Calculate the index for the percentile
    t = len(valid_latency_sorted) * fraction
    idx = int(t) if int(t) < len(valid_latency_sorted) else len(valid_latency_sorted) - 1

    # Return the latency at the desired percentile
    return valid_latency_sorted[idx].item()


def storeTimestampsToCsv(filename, ids, eventTimeStamps, arrivalTimeStamps, processedTimeStamps, counts):
    """
    Store the timestamps and IDs into a CSV file.

    Args:
        ids: numpy array of IDs.
        eventTimeStamps: numpy array of event timestamps.
        arrivalTimeStamps: numpy array of arrival timestamps.
        processedTimeStamps: numpy array of processed timestamps.
    """
    # Create a DataFrame with the timestamps and ids
    df = pd.DataFrame({
        'id': ids,
        'eventTime': eventTimeStamps,
        'arrivalTime': arrivalTimeStamps,
        'processedTime': processedTimeStamps
    })
    import os
    head, tail = os.path.split(filename)
    if not os.path.isdir(head):
        os.makedirs(head)
    filename = filename +f"_{counts}_timestamps.csv"
    df.to_csv(filename, index=False)

    print(f"Timestamps saved to {filename}")
    

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
    
        
    def run_task(algo, ds, distance, count, run_count, search_type, private_query, runbook, write_ratio=0.5, num_threads=16, batch_size=100):
        search_results = []
        search_times = []
        result_map = {}
        num_searches = 0

        Q = ds.get_queries() if not private_query else ds.get_private_queries()
        print(fr"Got {Q.shape[0]} queries")  

        num_searches = 0
        
        num_write_threads = max(1, int(num_threads * write_ratio))
        num_read_threads = max(1, num_threads - num_write_threads)
        print(f"Total Threads: {num_threads} → {num_write_threads} Write Threads, {num_read_threads} Read Threads (write_ratio={write_ratio})")
        
        task_queue = queue.Queue()
        
        for step, entry in enumerate(runbook):
            match entry['operation']:
                case 'insert_and_search':
                    start = entry['start']
                    end = entry['end']
                    
                    for batch_start in range(start, end, batch_size):
                        batch_end = min(batch_start + batch_size, end)
                        batch_ids = np.arange(batch_start, batch_end, dtype=np.uint32)
                        
                        task_queue.put(("insert_and_s", batch_start, batch_end, batch_ids))
        
                case _:
                    raise NotImplementedError(f"Operation '{entry['operation']}' not supported.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            while not task_queue.empty():
                write_futures = []
                
                for _ in range(min(batch_size, task_queue.qsize())):
                    try:
                        task_type, batch_start, batch_end, batch_ids = task_queue.get(timeout=1)
                    except queue.Empty:
                        break
                  
                    if task_type == 'insert_and_search':
                        future = executor.submit(algo.insert, ds.get_data_in_range(batch_start, batch_end), batch_ids)
                        write_futures.append(future)
                        print(f"Submitted Insert Task: {batch_start} to {batch_end}")
                        
                search_futures = [executor.submit(algo.query, Q, count) for _ in range(num_read_threads)]
                print(f"Submitted {num_read_threads} Search Tasks")
                
                for search_future in search_futures:
                    search_result = search_future.result()
                    search_results.append(search_result)
                    num_searches += 1
                    
                print(f"Completed {num_read_threads} Queries after {batch_end} insertions.")
                
        # recall_accuracy = algo.evaluate_recall_accuracy(search_results)
        # print(f"Final Recall Accuracy: {recall_accuracy:.4f}")
        
        attrs = {
            "name": str(algo),
            "run_count": run_count,
            "distance": distance,
            "type": search_type,
            "count": int(count),
            "search_times": search_times,
            "num_searches": num_searches,
            "private_queries": private_query,
            # "recall_accuracy": recall_accuracy,  
        }

        # print(f"Final Accuracy: {recall_accuracy * 100:.2f}%")
        return attrs, search_results