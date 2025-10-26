from __future__ import absolute_import

import itertools
from fileinput import filename

import numpy
import os
import json
import ast
import yaml
import random
import traceback
import re

import benchmark.concurrent
from benchmark.plotting.metrics import all_metrics as metrics, get_recall_values, get_recall_values_by_vecs
from benchmark.sensors.power_capture import power_capture
from benchmark.dataset_io import knn_result_read, knn_vec_result_read
import benchmark.streaming.compute_gt
import benchmark.congestion.compute_gt
import benchmark.concurrent.compute_gt
from benchmark.streaming.load_runbook import load_runbook_streaming
from benchmark.congestion.load_runbook import load_runbook_congestion
from benchmark.concurrent.load_runbook import load_runbook_concurrent

try:
    import PyCANDYAlgo
except ModuleNotFoundError:
    PyCANDYAlgo = None
import pandas as pd


_in_memory_metrics_cache = {}




def _parse_attr(value, default=None):
    if isinstance(value, (list, tuple, dict)):
        return value
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                return parsed
            except Exception:
                continue
    return default if default is not None else value


def _to_number(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return float(default)


class InMemoryGroup(dict):
    def __init__(self):
        super().__init__()
        self.attrs = {}

    def create_group(self, name):
        group = InMemoryGroup()
        self[name] = group
        return group

    def __contains__(self, key):
        return dict.__contains__(self, key) or key in self.attrs

    def __getitem__(self, key):
        return dict.__getitem__(self, key)


def get_or_create_metrics(run, cache_key=None, read_only=False):
    if 'metrics' in run:
        return run['metrics']
    if not read_only:
        try:
            return run.create_group('metrics')
        except Exception:
            pass
    if cache_key is None:
        cache_key = getattr(getattr(run, 'file', None), 'filename', None) or id(run)
    return _in_memory_metrics_cache.setdefault(cache_key, InMemoryGroup())


def create_pointset(data, xn, yn):
    xm, ym = (metrics[xn], metrics[yn])
    rev_y = -1 if ym["worst"] < 0 else 1
    rev_x = -1 if xm["worst"] < 0 else 1
    data.sort(key=lambda t: (rev_y * t[-1], rev_x * t[-2]))

    axs, ays, als = [], [], []
    # Generate Pareto frontier
    xs, ys, ls = [], [], []
    last_x = xm["worst"]
    comparator = ((lambda xv, lx: xv > lx)
                  if last_x < 0 else (lambda xv, lx: xv < lx))
    for algo, algo_name, xv, yv in data:
        if not xv or not yv:
            continue
        axs.append(xv)
        ays.append(yv)
        als.append(algo_name)
        if comparator(xv, last_x):
            last_x = xv
            xs.append(xv)
            ys.append(yv)
            ls.append(algo_name)
    return xs, ys, ls, axs, ays, als


def compute_metrics(true_nn, res, metric_1, metric_2,
                    recompute=False):
    all_results = {}
    for i, (properties, run) in enumerate(res):
        algo = properties['algo']
        algo_name = properties['name']
        # cache indices to avoid access to hdf5 file
        if metric_1 == "ap"  or metric_2 == "ap":
            run_nn = (numpy.array(run['lims']),
                    numpy.array(run['neighbors']),
                    numpy.array(run['distances']))
        else:
            run_nn = numpy.array(run['neighbors'])
        if recompute and 'metrics' in run:
            del run['metrics']
        if recompute:
            _in_memory_metrics_cache.pop(properties.get('filename'), None)
        metrics_cache = get_or_create_metrics(run, properties.get('filename'), properties.get('_read_only', False))

        metric_1_value = metrics[metric_1]['function'](
            true_nn, run_nn, metrics_cache, properties)
        metric_2_value = metrics[metric_2]['function'](
            true_nn, run_nn, metrics_cache, properties)

        print('%3d: %80s %12.3f %12.3f' %
              (i, algo_name, metric_1_value, metric_2_value))

        all_results.setdefault(algo, []).append(
            (algo, algo_name, metric_1_value, metric_2_value))

    return all_results


def compute_metrics_all_runs(dataset, dataset_name, res, recompute=False, 
        sensor_metrics=False, search_times=False,
        private_query=False, neurips23track=None, runbook_path=None):

    if neurips23track == 'congestion' and runbook_path:
        dataset_params = get_dataset_params_from_runbook(runbook_path, dataset_name)

    try:
        if neurips23track not in ['streaming', 'congestion', 'concurrent']:
            true_nn_across_steps = []
            true_nn = dataset.get_private_groundtruth() if private_query else dataset.get_groundtruth()
        elif neurips23track == 'streaming':
            true_nn_across_steps = []
            gt_dir = benchmark.streaming.compute_gt.gt_dir(dataset, runbook_path)
            max_pts, runbook = load_runbook_streaming(dataset_name, dataset.nb, runbook_path)
            for step, entry in enumerate(runbook):
                if entry['operation'] == 'search':
                    step_gt_path = os.path.join(gt_dir, 'step' + str(step+1) + '.gt100')
                    true_nn = knn_result_read(step_gt_path)
                    true_nn_across_steps.append(true_nn)
        elif neurips23track == "concurrent":
            true_nn_across_steps = []
            gt_dir = benchmark.concurrent.compute_gt.gt_dir(dataset, runbook_path)
            max_pts, cc_config, runbook = load_runbook_concurrent(dataset_name, dataset.nb, runbook_path)
            for step, entry in enumerate(runbook):
                if entry['operation'] == 'search':
                    step_gt_path = os.path.join(gt_dir, 'step' + str(step+1) + '.gt100')
                    true_nn = knn_vec_result_read(step_gt_path)
                    true_nn_across_steps.append(true_nn)
        elif neurips23track == "congestion":
            true_nn_across_steps = []
            gt_dir = benchmark.congestion.compute_gt.gt_dir(dataset, runbook_path)
            max_pts, runbook = load_runbook_congestion(dataset_name, dataset.nb, runbook_path)
            true_nn_across_batches = []
            num_batch_insert = 0
            for step, entry in enumerate(runbook):
                if entry['operation'] == 'search':
                    if neurips23track == 'congestion':
                        step_gt_path = os.path.join(gt_dir, 'step' + str(step) + '.gt100')
                    else:
                        step_gt_path = os.path.join(gt_dir, 'step' + str(step+1) + '.gt100')
                    if not os.path.exists(step_gt_path):
                        print(f"Warning: ground truth file missing for step {step}: {step_gt_path}")
                    else:
                        true_nn = knn_result_read(step_gt_path)
                        true_nn_across_steps.append(true_nn)

                if entry['operation'] == 'batch_insert' or entry['operation'] == 'batch_insert_delete':
                    batchSize = entry['batchSize']
                    temp_gt_dir = gt_dir
                    if batchSize==2500 and runbook_path!='neurips23/runbooks/congestion/test_experiment.yaml' and "batchDeletion" not in runbook_path and "bulkDeletion" not in runbook_path and "conceptDrift" not in runbook_path and "dimensions" not in runbook_path and "multiModal" not in runbook_path and "wordContamination" not in runbook_path:
                        temp_gt_dir = benchmark.congestion.compute_gt.gt_dir(dataset, 'neurips23/runbooks/congestion/general_experiment/general_experiment.yaml')

                    true_nn_across_batches.append([])
                    end = entry['end']
                    start = entry['start']
                    batchSize = entry['batchSize']
                    batch_step = (end - start) // batchSize
                    continuous_counter = 0
                    for i in range(batch_step):
                        continuous_counter += batchSize
                        if(continuous_counter >= (end-start)/100):
                            step_gt_path = os.path.join(temp_gt_dir, 'batch' +str(num_batch_insert) +"_"+str(i) + '.gt100')
                            if not os.path.exists(step_gt_path):
                                print(f"Warning: ground truth file missing for batch {num_batch_insert}_{i}: {step_gt_path}")
                            else:
                                true_nn = knn_result_read(step_gt_path)
                                true_nn_across_batches[-1].append(true_nn)
                            continuous_counter = 0
                    if (start + batch_step * batchSize < end and start + (batch_step + 1) * batchSize > end):
                        continuous_counter += batchSize
                        if(continuous_counter>=(end-start)/100):
                            step_gt_path = os.path.join(temp_gt_dir, 'batch' + str(num_batch_insert) + "_" + str(batch_step) + '.gt100')
                            if not os.path.exists(step_gt_path):
                                print(f"Warning: ground truth file missing for batch {num_batch_insert}_{batch_step}: {step_gt_path}")
                            else:
                                true_nn = knn_result_read(step_gt_path)
                                true_nn_across_batches[-1].append(true_nn)
                    num_batch_insert += 1
                    
    except:
        print(f"Groundtruth for {dataset} not found.")
        traceback.print_exc()
        return
    
    search_type = dataset.search_type()
    for i, (properties, run) in enumerate(res):
        algo = properties['algo']
        algo_name = properties['name']
        continuous_results_attr = _parse_attr(properties.get('continuousQueryResults'), [])
        continuous_latencies_attr = _parse_attr(properties.get('continuousQueryLatencies'), [])
        latency_insert_attr = _parse_attr(properties.get('latencyInsert'), [])
        insert_throughput_attr = _parse_attr(properties.get('insertThroughput'), [])
        latency_query_attr = _parse_attr(properties.get('latencyQuery'), [])
        query_size = _to_number(properties.get('querySize', 0), 0)
        properties['count'] = int(_to_number(properties.get('count', 0), 0))
        num_searches_value = int(_to_number(properties.get('num_searches', 0), 0))
        properties['num_searches'] = num_searches_value
        # cache distances to avoid access to hdf5 file
        if search_type == "knn" or search_type == "knn_filtered":
            if neurips23track in ['streaming', 'congestion']:
                run_nn_across_steps = []
                run_nn_across_batches = []
                for idx in range(num_searches_value):
                    step_suffix = str(properties['step_' + str(idx)])
                    run_nn_across_steps.append(numpy.array(run['neighbors_step' + step_suffix]))
                if isinstance(continuous_results_attr, list):
                    for batch in continuous_results_attr:
                        batch_arrays = []
                        if isinstance(batch, list):
                            for entry in batch:
                                batch_arrays.append(numpy.array(entry))
                        if batch_arrays:
                            run_nn_across_batches.append(batch_arrays)
            elif neurips23track == 'concurrent':
                run_nn_across_steps = []
                run_nn_across_batches = []
                for i in range(0, properties['num_searches']):
                   step_suffix = str(properties['step_' + str(i)])
                   run_nn_across_steps.append(numpy.array(run['neighbors_step' +  step_suffix]))
            else:
                run_nn = numpy.array(run['neighbors'])
        elif search_type == "range":
            if neurips23track in ['streaming', 'congestion', 'concurrent']:
                run_nn_across_steps = []
                for i in range(1,run['num_searches']):
                    step_suffix = str(properties['step_' + str(i)])
                    run_nn_across_steps.append(
                        (
                        numpy.array(run['neighbors_step' + step_suffix]),
                        numpy.array(run['neighbors_step' + step_suffix]),
                        numpy.array(run['distances_step' + step_suffix])
                        )
                    )
            else:
                run_nn = (numpy.array(run['lims']),
                        numpy.array(run['neighbors']),
                        numpy.array(run['distances']))
        if recompute and 'metrics' in run:
            print('Recomputing metrics, clearing cache')
            del run['metrics']
        if recompute:
            _in_memory_metrics_cache.pop(properties.get('filename'), None)
        metrics_cache = get_or_create_metrics(run, properties.get('filename'), properties.get('_read_only', False))

        dataset = properties['dataset']
        try:
            dataset = dataset.decode()
            algo = algo.decode()
            algo_name = algo_name.decode()
        except:
            pass

        dataset_info = dataset
        if neurips23track == 'congestion' and dataset_params:
            batchsize = dataset_params.get('batchSize', None)
            eventrate = dataset_params.get('eventRate', None)
            if batchsize and eventrate:
                dataset_info = f"{dataset}(BatchSize: {batchsize}, EventRate: {eventrate})"

        run_result = {
            'algorithm': algo,
            'parameters': algo_name,
            'dataset': dataset_info if neurips23track not in ['streaming', 'congestion', 'concurrent'] 
                        else dataset_info + '(' + os.path.split(runbook_path)[-1] + ')',
            'count': properties['count'],
        }
        for name, metric in metrics.items():
            if search_type == "knn" and name == "ap" or\
                search_type == "range" and name == "k-nn" or\
                search_type == "knn_filtered" and name == "ap" or\
                neurips23track in ["streaming", 'congestion', 'concurrent'] and name == "qps" or\
                neurips23track in ["streaming", 'congestion', 'concurrent'] and name == "queriessize":
                continue
            if not sensor_metrics and name=="wspq": #don't process power sensor_metrics by default
                continue
            if not search_times and name=="search_times": #don't process search_times by default
                continue
            if neurips23track in ['streaming', 'congestion']:
                v = []
                bv = []
                if len(true_nn_across_steps) != len(run_nn_across_steps):
                    print("Warning: mismatch between ground-truth steps (%d) and results (%d); truncating to common length." % (len(true_nn_across_steps), len(run_nn_across_steps)))
                for true_nn, run_nn in zip(true_nn_across_steps, run_nn_across_steps):
                    if 'knn' in metrics_cache:
                        del metrics_cache['knn']
                    properties["use_vec"] = False
                    val = metric["function"](true_nn, run_nn, metrics_cache, properties)
                    v.append(val)
                if name == 'k-nn':
                    print('Recall: ', v)

                if len(true_nn_across_batches) != len(run_nn_across_batches):
                    print("Warning: mismatch between ground-truth batches (%d) and results (%d); truncating to common length." % (len(true_nn_across_batches), len(run_nn_across_batches)))
                for true_batches, run_batches in zip(true_nn_across_batches, run_nn_across_batches):
                    if len(true_batches) != len(run_batches):
                        print("Warning: batch length mismatch (%d vs %d); truncating." % (len(true_batches), len(run_batches)))
                    batch_values = []
                    for t, r in zip(true_batches, run_batches):
                        if r is None:
                            print("Warning: skipping run batch with None result.")
                            continue
                        if not hasattr(r, '__len__'):
                            print("Warning: skipping run batch with unsized result (%r)." % (r,))
                            continue
                        if hasattr(t, '__len__') and len(t) == 0:
                            print("Warning: skipping ground-truth batch with no entries.")
                            continue
                        try:
                            mean, std, recalls, queries_with_ties = get_recall_values(t, r, properties['count'])
                        except Exception as exc:
                            print("Warning: failed to compute batch recall (%s); skipping." % exc)
                            continue
                        batch_values.append(mean)
                    if batch_values:
                        bv.append(batch_values)
            elif neurips23track == 'concurrent':
                v = []
                bv = []
                assert len(true_nn_across_steps) == len(run_nn_across_steps)
                for (true_nn, run_nn) in zip(true_nn_across_steps, run_nn_across_steps):
                    clear_cache = True

                    if clear_cache and 'knn' in metrics_cache:
                        del metrics_cache['knn']
                    
                    if name == 'k-nn':
                        properties["use_vec"] = True
                        val = metric["function"](true_nn, run_nn, metrics_cache, properties)
                        v.append(val)
                        print('Recall: ', v)
                
            else:
                v = metric["function"](true_nn, run_nn, metrics_cache, properties)

            if name=="k-nn":
                if neurips23track != "concurrent":
                    for i in range(len(v)):
                        run_result['knn_'+str(i)] = v[i]

                    long_data = []
                    for batch_idx, batch_values in enumerate(bv):
                        for value_idx, value in enumerate(batch_values):
                            long_data.append({
                                'batch_id': batch_idx,
                                'value_index': value_idx,
                                'value': value,
                            })

                    tag = properties['updateMemoryFootPrint']
                    if long_data:
                        df = pd.DataFrame(long_data)
                        # output_dir = "results_temp"
                        # os.makedirs(output_dir, exist_ok=True)
                        if neurips23track == 'congestion' and runbook_path:
                            last_part = runbook_path.split('/')[-1]
                            filename = os.path.join('results/neurips23/congestion', last_part)
                            f3_prefix = os.path.join(filename, f"{dataset_name}/{properties['count']}/{algo_name}")
                            os.makedirs(f3_prefix, exist_ok=True)
                            base_name = os.path.splitext(os.path.basename(properties["filename"]))[0]
                            f3 = os.path.join(
                                f3_prefix,
                                f"{base_name}_batch_recall.csv"
                            )
                            df.to_csv(f3, index=False)

                    for i in range(len(bv)):
                        recall_sum = 0
                        latency_sum = 0
                        for j in range(len(bv[i])):
                            recall_sum += bv[i][j]
                            if isinstance(continuous_latencies_attr, list) and i < len(continuous_latencies_attr):
                                latency_values = continuous_latencies_attr[i]
                                if isinstance(latency_values, list) and j < len(latency_values):
                                    latency_sum += _to_number(latency_values[j], 0)
                        if len(bv[i]):
                            run_result['continuousRecall_'+str(i)] = recall_sum/len(bv[i])
                            avg_latency = latency_sum/len(bv[i]) if len(bv[i]) else 0
                            if avg_latency > 0:
                                run_result['continuousLatency_'+str(i)] = avg_latency
                                if query_size:
                                    run_result['continuousThroughput_'+str(i)] = query_size/((avg_latency)/1e6)

                    if isinstance(latency_insert_attr, list):
                        for idx, value in enumerate(latency_insert_attr):
                            run_result['latencyInsert_'+str(idx)] = _to_number(value, 0)
                            if isinstance(insert_throughput_attr, list) and idx < len(insert_throughput_attr):
                                run_result['insertThroughput' + str(idx)] = _to_number(insert_throughput_attr[idx], 0)
                    if isinstance(latency_query_attr, list):
                        for idx, value in enumerate(latency_query_attr):
                            denom = _to_number(value, 0)
                            run_result['latencyQuery_'+str(idx)] = denom
                            if denom > 0 and query_size:
                                run_result['queryThroughput'+str(idx)] = query_size/(denom/1e6)


                    run_result['updateMemFPRT'] = _to_number(properties.get('updateMemoryFootPrint'), 0)
                    run_result['searchMemFPRT'] = _to_number(properties.get('searchMemoryFootPrint'), 0)
                    run_result['querySize'] = query_size
                else:
                    pass

            run_result[name] = numpy.nanmean(v)
        yield run_result


def compute_cc_metrics_all_runs(dataset, dataset_name, attrs, runbook_path=None):    
    # gt_dir = benchmark.concurrent.compute_gt.gt_dir(dataset, runbook_path)
    # max_pts, cc_config, runbook = load_runbook_concurrent(dataset_name, dataset.nb, runbook_path)

    # stepwise_res = []
    # stepwise_gt = {}
    # stepwise_recalls = []
    
    # for step, entry in enumerate(runbook):
    #     if entry['operation'] == 'insert_and_search':
    #         path = os.path.join(gt_dir, 'step' + str(step+1) + '.cc.gt.hdf5')
    #         parts = path.split('/')
    #         ds = parts[1].lower()
    #         bnum = re.search(r"batch(\d+)", parts[3]).group(1)
            
    #         parent_path = os.path.dirname(gt_dir)
    #         items = os.listdir(parent_path)
            
    #         step_gt_path = ""
    #         for root, dirs, files in os.walk(parent_path):
    #             for file in files:
    #                 if file.lower().endswith("cc.gt.hdf5"):
    #                     step_gt_path = os.path.join(root, file)
    #         stepwise_gt[f"{ds}{bnum}"] = step_gt_path
            
            
    metrics = []
    for properties, reader in attrs:
        row = next(reader)  
        # stepwise_res.append(row.get('cc_result_filename', None))
        
        # write_ratio_str = str(int(float(row.get('write_ratio', None)) * 100))
        # stepwise_recall_suffix = dataset_name + "_b" + row.get('batch_size', None)  \
        #     + "_w" + write_ratio_str + "_t" + row.get('num_threads', None)
        
        metrics.append({
            "insertThroughput": row.get('insert_throughput', None),  
            "searchThroughput": row.get('search_throughput', None),
            "insertLatencyAvg": row.get('insert_latency_avg', None),
            "searchLatencyAvg": row.get('search_latency_avg', None),
            "insertLatency95": row.get('insert_latency_95', None),
            "searchLatency95": row.get('search_latency_95', None),
            # "stepwiseRecallFile": stepwise_recall_suffix
        })
    
    # for res in stepwise_res:
    #     parts = res.split('/')
    #     ds = parts[4].lower()
    #     bnum = re.search(r"batch(\d+)", parts[3]).group(1)
        
    #     ds_key = f"{ds}{bnum}"
    #     gt = ""
    #     if ds_key in stepwise_gt.keys():
    #         gt = stepwise_gt[ds_key]
    #     else:
    #         print("Can't find gt for ", res)
        
    #     print("stepwise recall calculating : ", res, gt)
    #     stepwise_recalls.append(PyCANDYAlgo.calc_stepwise_recall(res, gt))

    # return metrics, stepwise_recalls\
    return metrics

def generate_n_colors(n):
    vs = numpy.linspace(0.3, 0.9, 7)
    colors = [(.9, .4, .4, 1.)]

    def euclidean(a, b):
        return sum((x - y)**2 for x, y in zip(a, b))
    while len(colors) < n:
        new_color = max(itertools.product(vs, vs, vs),
                        key=lambda a: min(euclidean(a, b) for b in colors))
        colors.append(new_color + (1.,))
    return colors


def create_linestyles(unique_algorithms):
    colors = dict(
        zip(unique_algorithms, generate_n_colors(len(unique_algorithms))))
    linestyles = dict((algo, ['--', '-.', '-', ':'][i % 4])
                      for i, algo in enumerate(unique_algorithms))
    markerstyles = dict((algo, ['+', '<', 'o', '*', 'x'][i % 5])
                        for i, algo in enumerate(unique_algorithms))
    faded = dict((algo, (r, g, b, 0.3))
                 for algo, (r, g, b, a) in colors.items())
    return dict((algo, (colors[algo], faded[algo],
                        linestyles[algo], markerstyles[algo]))
                for algo in unique_algorithms)


def get_up_down(metric):
    if metric["worst"] == float("inf"):
        return "down"
    return "up"


def get_left_right(metric):
    if metric["worst"] == float("inf"):
        return "left"
    return "right"


def get_plot_label(xm, ym):
    template = ("%(xlabel)s-%(ylabel)s tradeoff - %(updown)s and"
                " to the %(leftright)s is better")
    return template % {"xlabel": xm["description"],
                       "ylabel": ym["description"],
                       "updown": get_up_down(ym),
                       "leftright": get_left_right(xm)}


# Add the batchsize and eventrate parameters to the result file
def get_dataset_params_from_runbook(runbook_path, dataset_name):

    with open(runbook_path, 'r') as file:
        runbook = yaml.safe_load(file)

    if dataset_name not in runbook:
        print(f"Dataset {dataset_name} not found in runbook.")
        return None

    dataset_info = runbook[dataset_name]

    batchsize = None
    eventrate = None

    for step, step_details in dataset_info.items():
        if isinstance(step_details, dict):
            batchsize = step_details.get('batchSize', None)
            eventrate = step_details.get('eventRate', None)
            if batchsize and eventrate:
                return {'batchSize': batchsize, 'eventRate': eventrate}

    return None
