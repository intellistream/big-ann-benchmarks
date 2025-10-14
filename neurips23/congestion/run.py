import numpy as np
import time
import yaml
import pandas as pd
from benchmark.algorithms.base_runner import BaseRunner
from benchmark.datasets import DATASETS
from benchmark.results import get_result_filename
import tracemalloc
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


def getLatencyPercentile(fraction: float, event_time: np.ndarray, processed_time: np.ndarray) -> int:
    """
    Calculate the latency percentile from event and processed time tensors.

    :param fraction: float - Percentile in the range 0 ~ 1
    :param event_time: torch.Tensor - int64 tensor of event arrival timestamps
    :param processed_time: torch.Tensor - int64 tensor of processed timestamps
    :return: int - The latency value at the specified percentile
    """
    valid_latency = (processed_time - event_time)[(processed_time >= event_time) & (processed_time != 0)]

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


def store_timestamps_to_csv(filename, ids, eventTimeStamps, arrivalTimeStamps, processedTimeStamps, counts):
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


@dataclass
class StressTestConfig:
    start: int
    end: int
    warmup_batch: int
    warmup_events: int
    ramp_initial_batch: int
    ramp_scale: float
    ramp_events: int
    search_events: int
    search_tol_pct: float
    steady_events: int
    steady_eps_pct: float
    steady_backoff_pct: float
    grace_events: int
    deadline_us: float
    query_ratio: float

    @classmethod
    def from_entry(cls, entry: dict) -> "StressTestConfig":
        return cls(
            start=int(entry['start']),
            end=int(entry['end']),
            warmup_batch=int(entry['warmup_batch']),
            warmup_events=int(entry['warmup_events']),
            ramp_initial_batch=max(int(entry['ramp_initial_batch']), 1),
            ramp_scale=float(entry['ramp_scale']),
            ramp_events=int(entry['ramp_events']),
            search_events=int(entry['search_events']),
            search_tol_pct=float(entry['search_tol_pct']),
            steady_events=int(entry['steady_events']),
            steady_eps_pct=float(entry['steady_eps_pct']),
            steady_backoff_pct=float(entry['steady_backoff_pct']),
            grace_events=max(int(entry['grace_events']), 0),
            deadline_us=float(entry['deadline_us']),
            query_ratio=float(entry['query_ratio'])
        )

    def has_deadline(self) -> bool:
        return self.deadline_us > 0


@dataclass
class StressTestWindowResult:
    batch_size: int
    drop_delta: int
    pending_len: int
    ingest_latency_us: float
    query_latencies_us: List[float]
    window_duration_us: float
    interval_us: Optional[float]
    congested: bool


class StressTestController:
    def __init__(self, algo, ds, queries, count, config: StressTestConfig, attrs: dict):
        self.algo = algo
        self.ds = ds
        self.queries = queries
        self.count = count
        self.config = config
        self.attrs = attrs

        self.cursor = config.start
        self.query_carryover = 0.0
        self.prev_window_start: Optional[float] = None
        self.window_intervals: List[float] = []
        self.pending_consecutive = 0
        self.events_run = 0
        self.total_queries_run = 0
        self.best_metrics = None
        self.delta_hat_us: Optional[float] = None

    def run(self) -> dict:
        if not hasattr(self.algo, 'get_drop_count_delta') or not hasattr(self.algo, 'get_pending_queue_len'):
            raise RuntimeError("Stress test requires congestion-aware algorithm interface exposing drop and pending metrics.")

        # Reset drop counter baseline
        try:
            self.algo.get_drop_count_delta()
        except Exception:
            pass

        # Warmup phase
        if self.config.warmup_events > 0 and self.config.warmup_batch > 0:
            self._evaluate_batch(self.config.warmup_batch, self.config.warmup_events, collect_metrics=False)
            try:
                self.algo.get_drop_count_delta()
            except Exception:
                pass
            self.pending_consecutive = 0

        # Exponential ramp
        last_good = None
        first_bad = None
        batch = max(self.config.ramp_initial_batch, 1)
        ramp_iterations = 0
        while True:
            ramp_iterations += 1
            congested, windows = self._evaluate_batch(batch, self.config.ramp_events, collect_metrics=False)
            if not congested:
                last_good = batch
                self.best_metrics = self._summarize_windows(windows)
                next_batch = int(max(batch * self.config.ramp_scale, batch + 1))
                if next_batch <= batch:
                    next_batch = batch + 1
                if self.cursor + next_batch > self.config.end:
                    # Not enough data to continue ramping. Use the best observed batch.
                    break
                batch = next_batch
            else:
                first_bad = batch
                if last_good is None:
                    last_good = max(int(batch / self.config.ramp_scale), 1)
                break

            if self.cursor + batch > self.config.end:
                break

        if last_good is None:
            return {'stressTestStatus': 'failed', 'stressTestReason': 'No stable batch size observed during ramp.'}

        if first_bad is None:
            # Could not find congestion boundary; treat last good as B* candidate.
            b_star_candidate = last_good
        else:
            b_star_candidate = self._binary_search(last_good, first_bad)

        if b_star_candidate is None or b_star_candidate <= 0:
            return {'stressTestStatus': 'failed', 'stressTestReason': 'Binary search failed to converge on B*.'}

        steady_summary, final_batch = self._run_steady_phase(b_star_candidate)

        if steady_summary is None:
            return {
                'stressTestStatus': 'failed',
                'stressTestBStar': max(b_star_candidate, 0),
                'stressTestReason': 'Steady-state validation failed.'
            }

        delta_hat_us = steady_summary.get('interval_mean_us') or self.delta_hat_us
        if delta_hat_us is None or delta_hat_us <= 0:
            delta_hat_us = steady_summary.get('window_duration_mean_us')
        r_star = None
        if delta_hat_us and delta_hat_us > 0:
            r_star = final_batch / (delta_hat_us / 1e6)

        result = {
            'stressTestStatus': 'success',
            'stressTestBStar': final_batch,
            'stressTestRStar': r_star if r_star is not None else 0.0,
            'stressTestDeadlineUs': self.config.deadline_us,
            'stressTestDeltaHatUs': delta_hat_us if delta_hat_us else 0.0,
            'stressTestIngestP99Us': steady_summary.get('ingest_p99_us', 0.0),
            'stressTestIngestStdPct': steady_summary.get('ingest_std_pct', 0.0),
            'stressTestQueryP95Us': steady_summary.get('query_p95_us', 0.0),
            'stressTestQueryP99Us': steady_summary.get('query_p99_us', 0.0),
            'stressTestQueriesPerWindow': steady_summary.get('queries_per_window', 0.0),
            'stressTestWindowsObserved': steady_summary.get('window_count', 0),
            'stressTestEventsTotal': self.events_run,
            'stressTestTotalQueries': self.total_queries_run,
        }

        return result

    def _binary_search(self, last_good: int, first_bad: int) -> Optional[int]:
        low = max(last_good, 1)
        high = max(first_bad, low + 1)
        best_batch = low
        tolerance_pct = self.config.search_tol_pct / 100.0

        while high - low > 1:
            mid = (low + high) // 2
            congested, windows = self._evaluate_batch(mid, self.config.search_events, collect_metrics=False)
            if not congested:
                low = mid
                best_batch = mid
                self.best_metrics = self._summarize_windows(windows)
            else:
                high = mid

            if low > 0 and high > 0 and ((high - low) / low) <= tolerance_pct:
                break

        return best_batch

    def _run_steady_phase(self, initial_batch: int):
        batch = max(initial_batch, 1)
        backoff = min(max(self.config.steady_backoff_pct, 0.01), 0.5)

        while batch > 0:
            congested, windows = self._evaluate_batch(batch, self.config.steady_events, collect_metrics=True)
            summary = self._summarize_windows(windows)

            if not congested and summary.get('ingest_std_pct', 0.0) <= self.config.steady_eps_pct:
                return summary, batch

            next_batch = int(batch * (1.0 - backoff))
            if next_batch == batch:
                next_batch = batch - 1
            batch = max(next_batch, 0)

        return None, None

    def _evaluate_batch(self, batch_size: int, events: int, collect_metrics: bool) -> Tuple[bool, List[StressTestWindowResult]]:
        results: List[StressTestWindowResult] = []
        congested = False

        for _ in range(events):
            window_result = self._run_event_window(batch_size, collect_metrics)
            results.append(window_result)
            if window_result.congested:
                congested = True
                break

        return congested, results

    def _run_event_window(self, batch_size: int, collect_metrics: bool) -> StressTestWindowResult:
        if self.cursor + batch_size > self.config.end:
            raise RuntimeError(f"Dataset range exhausted while executing stress test window (cursor={self.cursor}, batch={batch_size}).")

        window_start = time.time()
        interval_us = None
        if self.prev_window_start is not None:
            interval_us = (window_start - self.prev_window_start) * 1e6
            self.window_intervals.append(interval_us)
            self.delta_hat_us = float(np.mean(self.window_intervals))
        self.prev_window_start = window_start

        ids = np.arange(self.cursor, self.cursor + batch_size, dtype=np.uint32)
        data = self.ds.get_data_in_range(self.cursor, self.cursor + batch_size)

        insert_start = time.time()
        self.algo.insert(data, ids)
        ingest_latency_us = (time.time() - insert_start) * 1e6
        self.cursor += batch_size

        total_queries_float = self.query_carryover + batch_size * self.config.query_ratio
        queries_this_window = int(total_queries_float)
        self.query_carryover = total_queries_float - queries_this_window

        query_latencies: List[float] = []
        for _ in range(queries_this_window):
            query_start = time.time()
            self.algo.query(self.queries, self.count)
            query_latency_us = (time.time() - query_start) * 1e6
            query_latencies.append(query_latency_us)
            try:
                self.algo.get_results()
            except AttributeError:
                pass
            self.total_queries_run += 1

        window_end = time.time()
        window_duration_us = (window_end - window_start) * 1e6

        drop_delta = self.algo.get_drop_count_delta()
        pending_len = self.algo.get_pending_queue_len()

        if pending_len > 0:
            self.pending_consecutive += 1
        else:
            self.pending_consecutive = 0

        deadline_us = self.config.deadline_us if self.config.has_deadline() else self.delta_hat_us
        if deadline_us is None or deadline_us <= 0:
            deadline_us = window_duration_us

        ingest_p99_us = ingest_latency_us  # Single sample surrogate

        pending_violation = self.pending_consecutive > self.config.grace_events
        drop_violation = drop_delta > 0
        ingestion_violation = ingest_p99_us > deadline_us

        congested = pending_violation or drop_violation or ingestion_violation

        self.events_run += 1

        return StressTestWindowResult(
            batch_size=batch_size,
            drop_delta=drop_delta,
            pending_len=pending_len,
            ingest_latency_us=ingest_latency_us,
            query_latencies_us=query_latencies if collect_metrics else [],
            window_duration_us=window_duration_us,
            interval_us=interval_us,
            congested=congested
        )

    def _summarize_windows(self, windows: List[StressTestWindowResult]) -> dict:
        if not windows:
            return {
                'window_count': 0,
                'ingest_p99_us': 0.0,
                'ingest_std_pct': 0.0,
                'query_p95_us': 0.0,
                'query_p99_us': 0.0,
                'queries_per_window': 0.0,
                'window_duration_mean_us': 0.0,
                'interval_mean_us': self.delta_hat_us or 0.0,
            }

        ingest_samples = np.array([w.ingest_latency_us for w in windows], dtype=float)
        query_samples = np.array([lat for w in windows for lat in w.query_latencies_us], dtype=float)
        window_durations = np.array([w.window_duration_us for w in windows], dtype=float)
        interval_samples = np.array([w.interval_us for w in windows if w.interval_us is not None], dtype=float)

        ingest_p99_us = float(np.percentile(ingest_samples, 99)) if ingest_samples.size else 0.0
        ingest_mean = float(np.mean(ingest_samples)) if ingest_samples.size else 0.0
        ingest_std_pct = 0.0
        if ingest_mean > 0 and ingest_samples.size > 1:
            ingest_std_pct = float((np.std(ingest_samples) / ingest_mean) * 100.0)

        query_p95_us = float(np.percentile(query_samples, 95)) if query_samples.size else 0.0
        query_p99_us = float(np.percentile(query_samples, 99)) if query_samples.size else 0.0

        interval_mean_us = float(np.mean(interval_samples)) if interval_samples.size else (self.delta_hat_us or 0.0)
        window_duration_mean_us = float(np.mean(window_durations)) if window_durations.size else 0.0

        queries_per_window = 0.0
        if windows:
            queries_per_window = sum(len(w.query_latencies_us) for w in windows) / len(windows)

        summary = {
            'window_count': len(windows),
            'ingest_p99_us': ingest_p99_us,
            'ingest_std_pct': ingest_std_pct,
            'query_p95_us': query_p95_us,
            'query_p99_us': query_p99_us,
            'queries_per_window': queries_per_window,
            'window_duration_mean_us': window_duration_mean_us,
            'interval_mean_us': interval_mean_us,
        }

        if interval_samples.size:
            self.delta_hat_us = interval_mean_us

        return summary


class CongestionRunner(BaseRunner):
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
    
    def run_task(algo, ds, distance, count, run_count, search_type, private_query, runbook, definition, query_arguments, runbook_path,dataset):
        best_search_time = float('inf')
        search_times = []
        all_results = []

        # data = ds.get_dataset()
        # ids = np.arange(1, ds.nb+1, dtype=np.uint32)

        Q = ds.get_queries() if not private_query else ds.get_private_queries()
        print(fr"Got {Q.shape[0]} queries")  

        # Load Runbook
        result_map = {}
        num_searches = 0
        num_batch = 0
        counts = {'initial':0,'batch_insert':0,'insert':0,'delete':0,'search':0,'stress_test':0}
        attrs = {
            "name": str(algo),
            "pendingWrite":0,
            "totalTime":0,
            "continuousQueryLatencies":[],
            "continuousQueryResults":[],
            'latencyInsert':[],
            'latencyQuery':[],
            'latencyDelete':[],
            'updateMemoryFootPrint':0,
            'searchMemoryFootPrint':0,
            'querySize':ds.nq,
            'insertThroughput':[],
            'batchLatency':[],
            'batchThroughput':[]
        }

        randomDrop = False
        randomContamination = False
        outOfOrder = False
        randomContaminationProb = 0.0
        randomDropProb = 0.0

        totalStart = time.time()
        for step, entry in enumerate(runbook):
            start_time = time.time()
            match entry['operation']:
                case 'initial':
                    start = entry['start']
                    end = entry['end']
                    ids = np.arange(start,end,dtype=np.uint32)
                    algo.initial_load(ds.get_data_in_range(start,end),ids)
                case 'startHPC':
                    print(type(algo))
                    algo.startHPC()
                case 'enableScenario':

                    if(entry.get("randomContamination", 0)==1):
                        randomContamination = True
                    if(entry.get("randomDrop", 0 )==1):
                        randomDrop = True
                    randomContaminationProb = entry.get("randomContaminationProb", 0.0)
                    randomDropProb = entry.get("randomDropProb",0.0)

                    if(entry.get("outOfOrder", 0 )==1):
                        outOfOrder = True

                    algo.enableScenario(randomContamination, randomContaminationProb, randomDrop, randomDropProb, outOfOrder)

                case 'endHPC':
                    algo.endHPC()
                case 'waitPending':
                    print("There is pending write: wait first")
                    t0 = time.time()
                    algo.waitPendingOperations()
                    attrs['pendingWrite'] += (time.time()-t0)*1e6
                    print('Pending write time: ')
                    print(attrs['pendingWrite'])
                case 'batch_insert':
                    tracemalloc.start()
                    start = entry['start']
                    end = entry['end']
                    batchSize = entry['batchSize']
                    eventRate = entry['eventRate']
                    print(f"Inserting with batch size={batchSize}")
                    batch_step = (end-start)//batchSize
                    ids = np.arange(start, end, dtype=np.uint32)
                    eventTimeStamps = generateTimestamps(rows=end-start,eventRate=eventRate)
                    arrivalTimeStamps = np.zeros(end-start,dtype=int)
                    processedTimeStamps = np.zeros(end-start, dtype=int)
                    attrs["latencyInsert"].append(0)
                    attrs['continuousQueryLatencies'].append([])
                    attrs['continuousQueryResults'].append([])



                    start_time = time.time()
                    continuous_counter = 0
                    MERGE_THRESHOLD = 5000000
                    inserted_total = 0
                    for i in range(batch_step):


                        data = ds.get_data_in_range(start+i*batchSize,start+(i+1)*batchSize)
                        insert_ids = ids[i*batchSize:(i+1)*batchSize]
                        if(randomContamination ):
                            if(random.random()<randomContaminationProb):
                                print(f"RANDOM CONTAMINATING DATA {ids[0]}:{ids[-1]}")
                                data = np.random.random(data.shape)

                        if(outOfOrder):
                            length = data.shape[0]
                            order = np.random.permutation(length)
                            temp_data = data
                            data = data[order]
                            insert_ids = insert_ids[order]

                        tNow = (time.time()-start_time)*1e6
                        tExpectedArrival = eventTimeStamps[(i+1)*batchSize-1]
                        while tNow<tExpectedArrival:
                            # busy waiting for a batch to arrive
                            tNow = (time.time()-start_time)*1e6
                        arrivalTimeStamps[i*batchSize:(i+1)*batchSize] = tExpectedArrival



                        #print(f'step {start+i*batchSize}:{start+(i+1)*batchSize}')





                        t0 = time.time()
                        algo.insert(data, insert_ids)
                        attrs["latencyInsert"][-1]+=(time.time()-t0)*1e6
                        processedTimeStamps[i*batchSize:(i+1)*batchSize] = (time.time()-start_time)*1e6
                        inserted_total += len(insert_ids)

                        #algo.waitPendingOperations()
                        # continuous query phase
                        continuous_counter += batchSize
                        if(continuous_counter >= (end-start)/100):
                            attrs['batchLatency'].append(0)
                            attrs['batchThroughput'].append(0)
                            print(f"{i}: {start + i * batchSize}~{start + (i + 1) * batchSize} querying")
                            t1 = time.time()
                            algo.query(Q, count)
                            attrs['continuousQueryLatencies'][-1].append((time.time() - t1) * 1e6)
                            attrs['batchLatency'][-1] += (time.time() - t1) * 1e6
                            querysize = Q.shape[0]
                            attrs['batchThroughput'][-1] += (querysize / ((attrs['batchLatency'][-1]) / 1e6))
                            results = algo.get_results()
                            attrs[f'continuousQueryResults'][-1].append(results)
                            #attrs[f'continuousQueryRecall{num_batch}_{i}'] = results
                            continuous_counter = 0



                        if inserted_total >= MERGE_THRESHOLD and algo.name == "freshdiskann":
                            print(f"MERGE THRESHOLD reached at {inserted_total} insertions — Performing merge()")
                            algo.final_merge()
                            inserted_total = 0

                    # process the rest
                    if(start+batch_step*batchSize<end and start+(batch_step+1)*batchSize>end):

                        tNow = (time.time()-start_time)*1e6
                        tExpectedArrival = eventTimeStamps[end-start-1]
                        while tNow<tExpectedArrival:
                            # busy waiting for a batch to arrive
                            tNow = (time.time()-start_time)*1e6

                        data = ds.get_data_in_range(start+batch_step*batchSize,end)
                        insert_ids = ids[batch_step*batchSize:]
                        if(randomContamination):
                            if(random.random()<randomContaminationProb):
                                print(f"RANDOM CONTAMINATING DATA {ids[0]}:{ids[-1]}")
                                data = np.random.random(data.shape)

                        if(outOfOrder):
                            length = data.shape[0]
                            order = np.random.permutation(length)
                            data = data[order]
                            insert_ids = insert_ids[order]


                        print(f'last {start+batch_step*batchSize}:{end}')
                        t0=time.time()



                        algo.insert(data, insert_ids)
                        attrs["latencyInsert"][-1]+=(time.time()-t0)*1e6
                        processedTimeStamps[batch_step*batchSize:end-start] = (time.time() - start_time) * 1e6
                        arrivalTimeStamps[batch_step*batchSize:end-start] = tExpectedArrival

                        #algo.waitPendingOperations()
                        # continuous query phase
                        continuous_counter += batchSize
                        if(continuous_counter >= (end-start)/100):
                            attrs['batchLatency'].append(0)
                            attrs['batchThroughput'].append(0)
                            print(f"{i}: {start + i * batchSize}~{end} querying")

                            t1 = time.time()
                            algo.query(Q, count)
                            attrs['continuousQueryLatencies'][-1].append((time.time() - t1) * 1e6)
                            attrs['batchLatency'][-1] += (time.time() - t1) * 1e6
                            attrs['batchThroughput'][-1] += ((end-start-batch_step*batchSize) / ((attrs['batchLatency'][-1]) / 1e6))
                            results = algo.get_results()
                            attrs['continuousQueryResults'][-1].append(results)
                            #attrs[f'continuousQueryRecall{num_batch}_{batch_step}'] = results
                            continuous_counter = 0


                    attrs['insertThroughput'].append((end-start)/((attrs['latencyInsert'][-1])/1e6))
                    filename = get_result_filename(dataset, count, definition, query_arguments, neurips23track="congestion", runbook_path=runbook_path)
                    store_timestamps_to_csv(filename, ids,eventTimeStamps, arrivalTimeStamps, processedTimeStamps, counts['batch_insert'])
                    counts['batch_insert'] +=1

                    current, peak = tracemalloc.get_traced_memory()
                    if peak>attrs['updateMemoryFootPrint']:
                        attrs['updateMemoryFootPrint'] = peak
                    tracemalloc.stop()

                    num_batch +=1

                case 'insert':
                    start = entry['start']
                    end = entry['end']
                    ids = np.arange(start, end, dtype=np.uint32)
                    algo.insert(ds.get_data_in_range(start, end), ids)

                    counts['insert'] +=1
                case 'delete':
                    ids = np.arange(entry['start'], entry['end'], dtype=np.uint32)
                    start = entry['start']
                    end = entry['end']
                    print(f'delete {start}:{end}')
                    algo.delete(ids)

                    counts['delete'] +=1
                case 'batch_insert_delete':
                    tracemalloc.start()
                    start = entry['start']
                    end = entry['end']
                    batchSize = entry['batchSize']
                    eventRate = entry['eventRate']
                    deletion_percentage = entry['deletion_percentage']
                    print(f"Inserting with batch size={batchSize}")
                    batch_step = (end - start) // batchSize
                    ids = np.arange(start, end, dtype=np.uint32)
                    eventTimeStamps = generateTimestamps(rows=end - start, eventRate=eventRate)
                    arrivalTimeStamps = np.zeros(end - start, dtype=int)
                    processedTimeStamps = np.zeros(end - start, dtype=int)
                    attrs["latencyInsert"].append(0)
                    attrs['continuousQueryLatencies'].append([])
                    attrs['continuousQueryResults'].append([])

                    start_time = time.time()
                    continuous_counter = 0
                    for i in range(batch_step):

                        data = ds.get_data_in_range(start + i * batchSize, start + (i + 1) * batchSize)
                        insert_ids = ids[i * batchSize:(i + 1) * batchSize]
                        if (randomContamination):
                            if (random.random() < randomContaminationProb):
                                print(f"RANDOM CONTAMINATING DATA {ids[0]}:{ids[-1]}")
                                data = np.random.random(data.shape)

                        if (outOfOrder):
                            length = data.shape[0]
                            order = np.random.permutation(length)
                            temp_data = data
                            data = data[order]
                            insert_ids = insert_ids[order]

                        tNow = (time.time() - start_time) * 1e6
                        tExpectedArrival = eventTimeStamps[(i + 1) * batchSize - 1]
                        while tNow < tExpectedArrival:
                            # busy waiting for a batch to arrive
                            tNow = (time.time() - start_time) * 1e6
                        arrivalTimeStamps[i * batchSize:(i + 1) * batchSize] = tExpectedArrival

                        # print(f'step {start+i*batchSize}:{start+(i+1)*batchSize}')

                        t0 = time.time()
                        algo.insert(data, insert_ids)

                        deletion_ids = ids[(int)((i+1) * batchSize-batchSize*deletion_percentage):(i + 1) * batchSize]
                        algo.delete(deletion_ids)
                        attrs["latencyInsert"][-1] += (time.time() - t0) * 1e6
                        print(f'delete {deletion_ids[0]}:{deletion_ids[-1]}')

                        processedTimeStamps[i * batchSize:(i + 1) * batchSize] = (time.time() - start_time) * 1e6

                        # algo.waitPendingOperations()
                        # continuous query phase
                        continuous_counter += batchSize
                        if (continuous_counter >= (end - start) / 100):
                            print(f"{i}: {start + i * batchSize}~{start + (i + 1) * batchSize} querying")
                            t0 = time.time()
                            algo.query(Q, count)
                            attrs['continuousQueryLatencies'][-1].append((time.time() - t0) * 1e6)

                            results = algo.get_results()
                            attrs[f'continuousQueryResults'][-1].append(results)
                            # attrs[f'continuousQueryRecall{num_batch}_{i}'] = results
                            continuous_counter = 0

                        # process the rest
                    if (start + batch_step * batchSize < end and start + (batch_step + 1) * batchSize > end):
                        tNow = (time.time() - start_time) * 1e6
                        tExpectedArrival = eventTimeStamps[end - start - 1]
                        while tNow < tExpectedArrival:
                            # busy waiting for a batch to arrive
                            tNow = (time.time() - start_time) * 1e6

                        data = ds.get_data_in_range(start + batch_step * batchSize, end)
                        insert_ids = ids[batch_step * batchSize:]
                        if (randomContamination):
                            if (random.random() < randomContaminationProb):
                                print(f"RANDOM CONTAMINATING DATA {ids[0]}:{ids[-1]}")
                                data = np.random.random(data.shape)

                        if (outOfOrder):
                            length = data.shape[0]
                            order = np.random.permutation(length)
                            data = data[order]
                            insert_ids = insert_ids[order]

                        print(f'last {start + batch_step * batchSize}:{end}')
                        t0 = time.time()

                        algo.insert(data, insert_ids)


                        deletion_ids = ids[int(end - batchSize * deletion_percentage):]
                        algo.delete(deletion_ids)
                        attrs["latencyInsert"][-1] += (time.time() - t0) * 1e6
                        print(f'delete {deletion_ids[0]}:{deletion_ids[-1]}')
                        processedTimeStamps[batch_step * batchSize:end-start] = (time.time() - start_time) * 1e6
                        arrivalTimeStamps[batch_step * batchSize:end-start] = tExpectedArrival

                        # algo.waitPendingOperations()
                        # continuous query phase
                        continuous_counter += batchSize
                        if (continuous_counter >= (end - start) / 100):
                            print(f"{i}: {start + i * batchSize}~{end} querying")

                            t0 = time.time()
                            algo.query(Q, count)
                            attrs['continuousQueryLatencies'][-1].append((time.time() - t0) * 1e6)

                            results = algo.get_results()
                            attrs['continuousQueryResults'][-1].append(results)
                            # attrs[f'continuousQueryRecall{num_batch}_{batch_step}'] = results
                            continuous_counter = 0

                    attrs['insertThroughput'].append((end - start) / ((attrs['latencyInsert'][-1]) / 1e6))
                    filename = get_result_filename(dataset, count, definition, query_arguments, neurips23track="congestion",
                                                   runbook_path=runbook_path)
                    store_timestamps_to_csv(filename, ids, eventTimeStamps, arrivalTimeStamps, processedTimeStamps,
                                            counts['batch_insert'])
                    counts['batch_insert'] += 1

                    current, peak = tracemalloc.get_traced_memory()
                    if peak > attrs['updateMemoryFootPrint']:
                        attrs['updateMemoryFootPrint'] = peak
                    tracemalloc.stop()

                    num_batch += 1

                case 'stress_test':
                    config = StressTestConfig.from_entry(entry)
                    try:
                        controller = StressTestController(algo, ds, Q, count, config, attrs)
                        result = controller.run()
                        attrs.update(result)
                        counts['stress_test'] += 1
                    except Exception as exc:
                        attrs['stressTestStatus'] = 'failed'
                        attrs['stressTestReason'] = str(exc)
                        raise

                case 'replace':
                    tags_to_replace = np.arange(entry['tags_start'], entry['tags_end'], dtype=np.uint32)
                    ids_start = entry['ids_start']
                    ids_end = entry['ids_end']
                    algo.replace(ds.get_data_in_range(ids_start, ids_end), tags_to_replace)
                case 'search':
                    tracemalloc.start()
                    if search_type == 'knn':
                        t0=time.time()
                        algo.query(Q, count)
                        attrs['latencyQuery'].append((time.time()-t0)*1e6)
                        results = algo.get_results()
                    current, peak = tracemalloc.get_traced_memory()
                    if peak>attrs['searchMemoryFootPrint']:
                        attrs['searchMemoryFootPrint'] = peak
                    tracemalloc.stop()
                    #
                    # elif search_type == 'range':
                    #     algo.range_query(Q, count)
                    #     results = algo.get_range_results()
                    # else:
                    #     raise NotImplementedError(f"Search type {search_type} not available.")
                    all_results.append(results)
                    result_map[num_searches] = step + 1
                    num_searches += 1

                    counts['search'] +=1

                case _:
                    raise NotImplementedError('Invalid runbook operation.')
            step_time = (time.time() - start_time)
            print(f"Step {step+1} took {step_time}s.")
        attrs["totalTime"] = (time.time()-totalStart) * 1e6
        attrs["run_count"]=run_count
        attrs["distance"]=distance
        attrs["type"]= search_type,
        attrs["count"] =int(count)
        attrs["search_times"]= search_times
        attrs["num_searches"]= num_searches
        attrs["private_queries"]=private_query

        # record each search
        for k, v in result_map.items():
            attrs['step_' + str(k)] = v
        additional = algo.get_additional()
        for k in additional:
            attrs[k] = additional[k]
        return (attrs, all_results)
