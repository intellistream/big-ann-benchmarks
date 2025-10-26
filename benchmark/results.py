from __future__ import absolute_import

import h5py
import json
import os
import re
import csv
import torch
import traceback
import pandas as pd

def store_latency(f1, f2, f3, attrs):
    latency_data = []
    throughput_data = []
    insert_latency_data = []

    # batchLatency
    for op_idx, op_latencies in enumerate(attrs.get('batchLatency', [])):
        for batch_idx, latency in enumerate(op_latencies[:-1]):
            latency_data.append({
                'op_id': op_idx,
                'batch_id': batch_idx,
                'batchLatency': latency,
            })

    # batchThroughput
    for op_idx, op_throughputs in enumerate(attrs.get('batchThroughput', [])):
        for batch_idx, throughput in enumerate(op_throughputs):
            throughput_data.append({
                'op_id': op_idx,
                'batch_id': batch_idx,
                'batchThroughput': throughput,
            })

    # batchInsertLatency
    for op_idx, op_insert_latencies in enumerate(attrs.get('batchinsertThroughtput', [])):
        for batch_idx, insert_latency in enumerate(op_insert_latencies):
            insert_latency_data.append({
                'op_id': op_idx,
                'batch_id': batch_idx,
                'batchinsertThroughtput': insert_latency,
            })

    if latency_data:
        pd.DataFrame(latency_data).to_csv(f1, index=False)

    if throughput_data:
        pd.DataFrame(throughput_data).to_csv(f2, index=False)

    if insert_latency_data:
        pd.DataFrame(insert_latency_data).to_csv(f3, index=False)


def get_result_filename(dataset=None, count=None, definition=None,
                        query_arguments=None, neurips23track=None, runbook_path=None):
    d = ['results']
    if neurips23track and neurips23track != 'none':
        d.append('neurips23')
        d.append(neurips23track)
        if neurips23track in ['streaming', 'congestion', 'concurrent']:
            if runbook_path == None:
                raise RuntimeError('Need runbook_path to store results')
            else:
                d.append(os.path.split(runbook_path)[1])
    if dataset:
        d.append(dataset)
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm)
        build_args = definition.arguments
        try:
            for args in build_args:
                if type(args) == dict and 'indexkey' in args:
                    build_args = [args['indexkey']]
        except:
                pass
        data = build_args + query_arguments
        data = re.sub(r'\W+', '_', json.dumps(data, sort_keys=True)).strip('_')
        if len(data) > 150:
            data = data[-149:]
        d.append(data)

    return os.path.join(*d)


def get_cc_result_filename(dataset=None, count=None, definition=None,
                            query_arguments=None, neurips23track=None, 
                            runbook_path=None, cc_config={}):
    d = ['results']
    if neurips23track and neurips23track != 'none':
        d.append('neurips23')
        d.append(neurips23track)
        if neurips23track in ['streaming', 'congestion', 'concurrent']:
            if runbook_path == None:
                raise RuntimeError('Need runbook_path to store results')
            else:
                d.append(os.path.split(runbook_path)[1])
    if dataset:
        d.append(dataset)
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm)
        build_args = definition.arguments
        try:
            for args in build_args:
                if type(args) == dict and 'indexkey' in args:
                    build_args = [args['indexkey']]
        except:
                pass
        data = build_args + query_arguments
        data = re.sub(r'\W+', '_', json.dumps(data, sort_keys=True)).strip('_')
        
        data += f"_b{cc_config['batch_size']}"
        data += f"_t{cc_config['num_threads']}"
            
        write_ratio_suffix = int(cc_config['write_ratio'] * 100)
        read_ratio_suffix = int((1 - cc_config['write_ratio']) * 100)
        data += f"_w{write_ratio_suffix}r{read_ratio_suffix}"
        
        if len(data) > 150:
            data = data[-149:]
        d.append(data)

    return os.path.join(*d)


def add_cc_results_to_h5py(f, search_type, results, count, suffix=''):
    if search_type in ["knn", "knn_filtered"]:
        if isinstance(results, list) and all(isinstance(x, torch.Tensor) for x in results):
            results = torch.stack(results)  
        results = results.numpy()
        results = results.T  
        f.create_dataset('neighbors' + suffix, results.shape, 'i', data=results)
    elif search_type == "range":
        lims, D, I = results
        f.create_dataset('neighbors' + suffix, data=I)
        f.create_dataset('lims' + suffix, data=lims)
        f.create_dataset('distances' + suffix, data=D)
    else:
        raise NotImplementedError(f"Search type '{search_type}' not supported.")


def add_results_to_h5py(f, search_type, results, count, suffix = ''):
    if search_type in ["knn", "knn_filtered"]:
        neighbors = f.create_dataset('neighbors' + suffix, (len(results), count), 'i', data = results)
    elif search_type == "range":
        lims, D, I= results
        f.create_dataset('neighbors' + suffix, data=I)
        f.create_dataset('lims' + suffix, data=lims)
        f.create_dataset('distances' + suffix, data=D)
    else:
        raise NotImplementedError()


def _normalize_attr_for_storage(value):
    import json
    import numpy as np

    def convert(obj):
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, (np.ndarray,)):
            return convert(obj.tolist())
        if isinstance(obj, (list, tuple, set)):
            return [convert(o) for o in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    simple = convert(value)
    if simple is None:
        return ''
    if isinstance(simple, (str, bytes, int, float, bool)):
        return simple
    try:
        return json.dumps(simple)
    except TypeError:
        return str(simple)


def store_results(dataset, count, definition, query_arguments,
        attrs, results, search_type, neurips23track=None, runbook_path=None, cc_config={}):
    
    fn = ""
    fn_attr = ""
    
    if neurips23track != "concurrent":
        fn = get_result_filename(
            dataset, count, definition, query_arguments, neurips23track, runbook_path) + '.hdf5'
        fn_attr = get_result_filename(
            dataset, count, definition, query_arguments, neurips23track, runbook_path) + '.csv'
    else:
        fn = get_cc_result_filename(
            dataset, count, definition, query_arguments, neurips23track, runbook_path, cc_config) + '.hdf5'
        fn_attr = get_cc_result_filename(
            dataset, count, definition, query_arguments, neurips23track, runbook_path, cc_config) + '.csv'
        
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)
    f = h5py.File(name=fn, mode='w', libver='latest')
    import pandas as pd
    safe_attrs = {k: _normalize_attr_for_storage(v) for k, v in attrs.items()}
    df = pd.DataFrame([safe_attrs])

    # Write the DataFrame to a CSV file
    df.to_csv(fn_attr, index=False)

    def _attr_value(value):
        normalized = _normalize_attr_for_storage(value)
        if isinstance(normalized, (dict, list, tuple, set)):
            try:
                return json.dumps(normalized, ensure_ascii=False)
            except Exception:
                return str(normalized)
        return normalized

    for k, v in attrs.items():
        try:
            f.attrs[k] = _attr_value(v)
        except TypeError:
            f.attrs[k] = str(_attr_value(v))
    
    if neurips23track in ['streaming', 'congestion']:
        for i, step_results in enumerate(results):
            step = attrs['step_' + str(i)]
            # store labels
            add_results_to_h5py(f, search_type, step_results, count, '_step' + str(step))
    elif neurips23track == 'concurrent':
        for i, step_results in enumerate(results):
            step = attrs['step_' + str(i)]
            # store tensors
            add_cc_results_to_h5py(f, search_type, step_results, count, '_step' + str(step))
    else:
        add_results_to_h5py(f, search_type, results, count)

    f.close()


def load_all_results(dataset=None, count=None, neurips23track="congestion", runbook_path=None):
    """
    A generator for all result files.
    """
    for root, _, files in os.walk(get_result_filename(dataset, count, \
                                                      neurips23track=neurips23track, \
                                                      runbook_path=runbook_path)):
        for fn in files:
            if os.path.splitext(fn)[-1] != '.hdf5':
                continue
            if fn.endswith('.cc.hdf5'):
                continue
            full_path = os.path.join(root, fn)
            print(f"Found HDF5 file: {full_path}")
            file_path = os.path.join(root, fn)
            read_only = False
            try:
                f = h5py.File(name=file_path, mode='r+', libver='latest', swmr=True)
            except OSError as exc:
                print(f"SKIP locked file for write: {full_path} ({exc})")
                try:
                    f = h5py.File(name=file_path, mode='r', libver='latest', swmr=True)
                    read_only = True
                except OSError as exc2:
                    print(f"Was unable to read {full_path}: {exc2}")
                    continue
            try:
                properties = dict(f.attrs)
                properties["filename"] = fn
                properties["_read_only"] = read_only
                yield properties, f
            finally:
                f.close()





def load_all_attrs(dataset=None, count=None, neurips23track="concurrent", runbook_path=None):
    for root, _, files in os.walk(get_result_filename(dataset, count, \
                                                      neurips23track=neurips23track, \
                                                      runbook_path=runbook_path)):
        for fn in files:
            if os.path.splitext(fn)[-1].lower() != '.csv':
                continue
            full_path = os.path.join(root, fn)
            print(f"Found CSV file: {full_path}")
            try:
                with open(full_path, mode='r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)  
                    properties = {"filename": fn}  
                    yield properties, reader  
            except Exception as e:
                print(f'Was unable to read {fn}')
                traceback.print_exc()
                

def get_unique_algorithms():
    return set(properties['algo'] for properties, _ in load_all_results())
