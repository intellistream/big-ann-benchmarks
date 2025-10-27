import argparse
import csv
import json
import math
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from benchmark.datasets import DATASETS
from benchmark.plotting.utils import compute_metrics_all_runs, compute_cc_metrics_all_runs
from benchmark.results import load_all_results, load_all_attrs

FAIRNESS_RUNBOOKS = ['neurips23/runbooks/congestion/fairness/fairness_static_10.yaml']
FAIRNESS_DATASETS = {'sift'}


def cleaned_run_metric(run_metrics):
    cleaned = []
    for run_metric in run_metrics:
        run_metric['track'] = track
        if 'k-nn' in run_metric:
            run_metric['recall/ap'] = run_metric['k-nn']
            del run_metric['k-nn']
        if 'ap' in run_metric:
            run_metric['recall/ap'] = run_metric['ap']
            del run_metric['ap']
        if args.sensors:
            if 'wspq' not in run_metric:
                print('Warning: wspq sensor data not available.')
        if args.search_times:
            search_times = run_metric['search_times'] 
            if 'search_times' in run_metric:
                # create a space separated list suitable as column for a csv
                run_metric['search_times'] = \
                    " ".join( [str(el) for el in search_times ] )

            if args.detect_caching != None:
                print("%s: Checking for response caching for these search times->" % dataset_name, search_times)
                percent_improvement = (search_times[0]-search_times[-1])/search_times[0]
                caching = percent_improvement > args.detect_caching
                run_metric['caching'] = "%d %f %f" % ( 1 if caching else 0, args.detect_caching, percent_improvement )
                if caching:
                    print("Possible caching discovered: %.3f > %.3f" % ( percent_improvement, args.detect_caching) )
                else:
                    print("No response caching detected.")

            else:
                print("Warning: 'search_times' not available.")
        cleaned.append(run_metric)
    return cleaned


def format_fairness_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Make the fairness export easier to read by renaming and reordering columns."""
    rename_map = {}
    # Rename step-wise recall metrics.
    for idx in range(10):
        rename_map[f'knn_{idx}'] = f'stepRecall_{idx + 1}'
        rename_map[f'continuousRecall_{idx}'] = f'continuousRecall_window_{idx + 1}'
        rename_map[f'continuousLatency_{idx}'] = f'continuousLatency_window_{idx + 1}'
        rename_map[f'continuousThroughput_{idx}'] = f'continuousThroughput_window_{idx + 1}'
        rename_map[f'latencyInsert_{idx}'] = f'insertLatency_stage_{idx + 1}'
        rename_map[f'insertThroughput{idx}'] = f'insertThroughput_stage_{idx + 1}'
        rename_map[f'latencyQuery_{idx}'] = f'queryLatency_stage_{idx + 1}'
        rename_map[f'queryThroughput{idx}'] = f'queryThroughput_stage_{idx + 1}'

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    primary_cols = [
        "algorithm",
        "parameters",
        "dataset",
        "track",
        "count",
        "recall/ap",
    ]
    step_cols = [f'stepRecall_{idx + 1}' for idx in range(10)]
    continuous_cols = []
    for prefix in ("continuousRecall", "continuousLatency", "continuousThroughput"):
        continuous_cols.extend([f'{prefix}_window_{idx + 1}' for idx in range(10)])

    insert_cols = []
    for prefix in ("insertLatency", "insertThroughput"):
        insert_cols.extend([f'{prefix}_stage_{idx + 1}' for idx in range(10)])

    query_cols = []
    for prefix in ("queryLatency", "queryThroughput"):
        query_cols.extend([f'{prefix}_stage_{idx + 1}' for idx in range(10)])

    events_series = df.get('maintenanceEvents')
    parsed_events = None
    max_events = 0
    if events_series is not None:
        def _parse_events(val):
            if pd.isna(val):
                return []
            if isinstance(val, (list, tuple)):
                return list(val)
            if isinstance(val, bytes):
                val = val.decode()
            if isinstance(val, str):
                val = val.strip()
                if not val:
                    return []
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return []
            return []

        parsed_events = events_series.apply(_parse_events)
        max_events = int(parsed_events.apply(len).max()) if len(parsed_events) else 0
    else:
        parsed_events = pd.Series([[] for _ in range(len(df))])
        max_events = 0

    if step_cols:
        step_values_array = df[step_cols].to_numpy()
    else:
        step_values_array = None

    rebuild_columns = []
    for event_idx in range(max_events):
        before_vals = []
        after_vals = []
        cost_vals = []
        for row_idx, events in enumerate(parsed_events):
            if event_idx < len(events):
                event = events[event_idx]
                step_number = event.get('step')
                after_pos = 0
                if isinstance(step_number, (int, float)):
                    after_pos = max(int(step_number) - 1, 0)
                if step_values_array is not None and step_values_array.shape[1] > 0:
                    after_pos = min(after_pos, step_values_array.shape[1] - 1)
                    before_pos = max(after_pos - 1, 0)
                    after_val = step_values_array[row_idx, after_pos]
                    before_val = step_values_array[row_idx, before_pos]
                else:
                    after_val = pd.NA
                    before_val = pd.NA
                before_vals.append(before_val)
                after_vals.append(after_val)
                cost_us = event.get('cost_us')
                if cost_us is None:
                    cost_vals.append(pd.NA)
                else:
                    try:
                        cost_vals.append(float(cost_us) / 1e6)
                    except (TypeError, ValueError):
                        cost_vals.append(pd.NA)
            else:
                before_vals.append(pd.NA)
                after_vals.append(pd.NA)
                cost_vals.append(pd.NA)
        base = f'rebuild{event_idx + 1}'
        before_col = f'{base}_recallBefore'
        after_col = f'{base}_recallAfter'
        cost_col = f'{base}_rebuildCostSeconds'
        df[before_col] = before_vals
        df[after_col] = after_vals
        df[cost_col] = cost_vals
        rebuild_columns.extend([before_col, after_col, cost_col])

    if 'maintenanceEvents' in df.columns:
        df = df.drop(columns=['maintenanceEvents'])

    ordered_cols = (
        [col for col in primary_cols if col in df.columns]
        + [col for col in step_cols if col in df.columns]
        + [col for col in rebuild_columns if col in df.columns]
        + [col for col in continuous_cols if col in df.columns]
        + [col for col in insert_cols if col in df.columns]
        + [col for col in query_cols if col in df.columns]
    )

    remaining_cols = [col for col in df.columns if col not in ordered_cols]

    # Return a new DataFrame with the columns arranged.
    return df[ordered_cols + remaining_cols]


def write_stepwise_recall_to_csv(stepwise_recall, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Step', 'Recall'])
        
        for step, recall in stepwise_recall:
            writer.writerow([step, recall])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        help='Path to the output csv file')
    parser.add_argument(
        '--track',
        choices=['streaming', 'congestion', 'concurrent'],
        required=True)
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Path to the output csv file')
    parser.add_argument(
        '--private-query',
        help='Use the private queries and ground truth',
        action='store_true')
    parser.add_argument(
        '--sensors',
        action='store_true',
        help='Export sensors data if available')
    parser.add_argument(
        '--search-times',
        action='store_true',
        help='Export search times data if available')
    parser.add_argument(
        '--detect-caching',
        type=float,
        default=None,
        metavar="THRESHOLD",
        help='Try to detect query response caching by analyzing search times.  Supply a threshold betwee 0 and 1, such as 0.3.')
    args = parser.parse_args()

    if args.detect_caching!=None and not args.search_times:
        print("Error: --detect_caching requires the --search_times flag")
        sys.exit(1)

    datasets = DATASETS.keys()
    dfs = []
    
    cc_dfs = []

    # neurips23tracks = ['streaming', 'congestion', 'concurrent', 'none']
    neurips23tracks = ['concurrent', 'none']
    tracks = [args.track]
    concurrent_dataset_name = ["reddit", "sift", "glove", "msong"]  
    stepwise_res_file = "stepwise"

    is_first = True
    for track in tracks:
        for dataset_name in datasets:
            if track == 'congestion' and args.output == 'fairness' and dataset_name not in FAIRNESS_DATASETS:
                continue
            if track=="concurrent" and dataset_name not in concurrent_dataset_name:
                continue
            print(f"Looking at track:{track}, dataset:{dataset_name}")
            dataset = DATASETS[dataset_name]()
            runbook_paths = [None]
            if track == 'streaming':
                runbook_paths = ['neurips23/runbooks/streaming/simple_runbook.yaml'
                                ]
            if track == 'concurrent':
                if not os.path.exists("stepwise"):
                    os.makedirs("stepwise")
                    
                runbook_paths = [
                                    'neurips23/runbooks/concurrent/batch100_w05r95.yaml',
                                    'neurips23/runbooks/concurrent/batch100_w10r90.yaml',
                                    'neurips23/runbooks/concurrent/batch100_w20r80.yaml',
                                    'neurips23/runbooks/concurrent/batch100_w50r50.yaml',
                                    'neurips23/runbooks/concurrent/batch100_w80r20.yaml',
                                    'neurips23/runbooks/concurrent/batch100_w90r10.yaml',
                                ]
            if track == 'congestion':
                runbook_paths = []
                if args.output == "fairness":
                    runbook_paths = FAIRNESS_RUNBOOKS
                if args.output == "gen":
                    runbook_paths = ['neurips23/runbooks/congestion/general_experiment/general_experiment.yaml'
                                    ]
                if args.output == "batch":
                    runbook_paths = ['neurips23/runbooks/congestion/batchSizes/batch100.yaml',
                                     'neurips23/runbooks/congestion/batchSizes/batch500.yaml',
                                     'neurips23/runbooks/congestion/batchSizes/batch1000.yaml',
                                     'neurips23/runbooks/congestion/batchSizes/batch2500.yaml',
                                     'neurips23/runbooks/congestion/batchSizes/batch5000.yaml',
                                     'neurips23/runbooks/congestion/batchSizes/batch10000.yaml',
                                     'neurips23/runbooks/congestion/batchSizes/batch20000.yaml',
                                     'neurips23/runbooks/congestion/batchSizes/batch50000.yaml',
                                     ]
                if args.output == "event":
                    runbook_paths = [
                                     'neurips23/runbooks/congestion/eventRates/event2500.yaml',
                                     'neurips23/runbooks/congestion/eventRates/event10000.yaml',
                                     'neurips23/runbooks/congestion/eventRates/event100000.yaml',
                                     'neurips23/runbooks/congestion/eventRates/event200000.yaml',
                                     'neurips23/runbooks/congestion/eventRates/event500000.yaml',
                                     ]
                if args.output=='conceptDrift':
                    runbook_paths=['neurips23/runbooks/congestion/conceptDrift/conceptDrift_experiment.yaml']
                if args.output=='randomContamination':
                    runbook_paths=['neurips23/runbooks/congestion/randomContamination/randomContamination0.05.yaml',
                                   'neurips23/runbooks/congestion/randomContamination/randomContamination0.10.yaml',
                                   'neurips23/runbooks/congestion/randomContamination/randomContamination0.15.yaml',
                                   'neurips23/runbooks/congestion/randomContamination/randomContamination0.20.yaml',
                                   'neurips23/runbooks/congestion/randomContamination/randomContamination0.25.yaml']
                if args.output == 'randomDrop':
                    runbook_paths=['neurips23/runbooks/congestion/randomDrop/randomDrop0.05.yaml',
                                   'neurips23/runbooks/congestion/randomDrop/randomDrop0.10.yaml',
                                   'neurips23/runbooks/congestion/randomDrop/randomDrop0.15.yaml',
                                   'neurips23/runbooks/congestion/randomDrop/randomDrop0.20.yaml',
                                   'neurips23/runbooks/congestion/randomDrop/randomDrop0.25.yaml']
                if args.output == 'wordContamination':
                    runbook_paths=['neurips23/runbooks/congestion/wordContamination/wordContamination_experiment.yaml']
                if args.output == 'bulkDeletion':
                    runbook_paths = ['neurips23/runbooks/congestion/bulkDeletion/bulkDeletion0.1.yaml',
                                     'neurips23/runbooks/congestion/bulkDeletion/bulkDeletion0.2.yaml',
                                     'neurips23/runbooks/congestion/bulkDeletion/bulkDeletion0.3.yaml',
                                     'neurips23/runbooks/congestion/bulkDeletion/bulkDeletion0.4.yaml',
                                     'neurips23/runbooks/congestion/bulkDeletion/bulkDeletion0.5.yaml']
                if args.output == 'batchDeletion':
                    runbook_paths = ['neurips23/runbooks/congestion/batchDeletion/batchDeletion0.1.yaml',
                                     'neurips23/runbooks/congestion/batchDeletion/batchDeletion0.2.yaml',
                                     'neurips23/runbooks/congestion/batchDeletion/batchDeletion0.3.yaml',
                                     'neurips23/runbooks/congestion/batchDeletion/batchDeletion0.4.yaml',
                                     'neurips23/runbooks/congestion/batchDeletion/batchDeletion0.5.yaml']
                if args.output == 'stressTest':
                    runbook_paths = ['neurips23/runbooks/congestion/stressTest/stressTest0.1.yaml',
                                     'neurips23/runbooks/congestion/stressTest/stressTest0.2.yaml',
                                     'neurips23/runbooks/congestion/stressTest/stressTest0.3.yaml',
                                     'neurips23/runbooks/congestion/stressTest/stressTest0.4.yaml',
                                     'neurips23/runbooks/congestion/stressTest/stressTest0.5.yaml']
                if args.output == "curseDim":
                    runbook_paths = ['neurips23/runbooks/congestion/dimensions/dimensions_experiment.yaml']
                if args.output == "multiModal":
                    runbook_paths = ['neurips23/runbooks/congestion/multiModal/multiModal_experiment.yaml']
                if args.output == "algoOpt":
                    runbook_paths = ['neurips23/runbooks/congestion/algo_optimizations/algo_optimizations.yaml']

            for runbook_path in runbook_paths:
                print("Looking for results", runbook_path)
                results = load_all_results(dataset_name, neurips23track=track, runbook_path=runbook_path)
                print("Looked results ", runbook_path)
                results = compute_metrics_all_runs(dataset, dataset_name, results, args.recompute, 
                    args.sensors, args.search_times, args.private_query, 
                    neurips23track=track, runbook_path=runbook_path)
                
                results = cleaned_run_metric(results)
                    
                if track == 'concurrent':
                    print("Looking for attrs ", runbook_path)
                    attrs = load_all_attrs(dataset_name, neurips23track=track, runbook_path=runbook_path)
                    print("Looked attrs ", runbook_path)
                    # cc_results, stepwise_recalls = compute_cc_metrics_all_runs(dataset, dataset_name, attrs, runbook_path=runbook_path)

                    # for i, (r, cc_r) in enumerate(zip(results, cc_results)):
                    #     new_name = r["algorithm"] + "_" + cc_r["stepwiseRecallFile"]
                    #     cc_r["stepwiseRecallFile"] = "stepwise/" + new_name + ".csv"
                    #     write_stepwise_recall_to_csv(stepwise_recalls[i], cc_r["stepwiseRecallFile"])
                    #     merged = r | cc_r
                    #     results[i] = {k: v for k, v in merged.items() if not (isinstance(v, float) and math.isnan(v))}
                        
                    # # for cc_r in cc_results:
                    # #     new_name = r["algorithm"] + "_" + cc_r["stepwiseRecallFile"]
                    # #     cc_r["stepwiseRecallFile"] = "stepwise/" + new_name + ".csv"
                    # #     write_stepwise_recall_to_csv(stepwise_recalls[i], cc_r["stepwiseRecallFile"])
                    
                    cc_results = compute_cc_metrics_all_runs(dataset, dataset_name, attrs, runbook_path=runbook_path)
                    
                    for i, (cc_r, r) in enumerate(zip(cc_results, results)):
                        merged = r | cc_r
                        results[i] = {k: v for k, v in merged.items() if not (isinstance(v, float) and math.isnan(v))}
                        
                if len(results) > 0:
                    dfs.append(pd.DataFrame(results))     

    dfs = [e for e in dfs if len(e) > 0]
    
    print(dfs)
    if len(dfs) > 0:
        data = pd.concat(dfs)
        sort_columns = ["algorithm", "dataset"]
        if "recall/ap" in data.columns:  
            sort_columns.append("recall/ap")
        
        if args.track == 'congestion' and args.output == 'fairness':
            data = format_fairness_dataframe(data)

        data = data.sort_values(by=sort_columns)
        
        if args.output is None:
            output_path = args.track + ".csv"
        else:
            output_path = f"{args.output}-{args.track}.csv"
        
        data.to_csv(output_path, index=False)
        print(output_path)
