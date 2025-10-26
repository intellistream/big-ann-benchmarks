import yaml
from typing import Any, Dict, List, Optional


class RunbookConfig(list):
    """List-like container with optional metadata for congestion runbooks."""

    def __init__(self, steps: List[dict], maintenance_policy: Optional[Dict[str, Any]] = None):
        super().__init__(steps)
        self.maintenance_policy: Dict[str, Any] = maintenance_policy or {}

def load_runbook_congestion(dataset_name, max_pts, runbook_file):


    with open(runbook_file) as fd:
        runbook = yaml.safe_load(fd)[dataset_name]
        i=1
        run_list = []
        while i in runbook:
            entry = runbook.get(i)
            if entry['operation'] not in {'initial','insert', 'delete', 'search', 'replace', 'batch_insert','startHPC', 'endHPC', 'waitPending', 'enableScenario', 'batch_insert_delete', 'auto_batch_delete', 'stress_test', 'maintenance_rebuild'}:
                raise Exception('Undefined runbook operation')
            if entry['operation'] in {'batch_insert', 'maintenance_rebuild', 'auto_batch_delete'}:
                if 'start' not in entry:
                    raise Exception('Start not speficied in runbook')
                if 'end' not in entry:
                    raise Exception('End not specified in runbook')
                if entry['operation'] in {'batch_insert', 'auto_batch_delete'} and 'batchSize' not in entry:
                    raise Exception('batchSize not specified in runbook')
            if entry['operation']  in {'initial','insert', 'delete', 'stress_test'}:
                if 'start' not in entry:
                    raise Exception('Start not speficied in runbook')
                if 'end' not in entry:
                    raise Exception('End not specified in runbook')
                if entry['start'] < 0 or entry['start'] >= max_pts:
                    raise Exception('Start out of range in runbook')
                if entry['end'] < 0 or entry['end'] > max_pts:
                    raise Exception('End out of range in runbook')
            if entry['operation'] == 'stress_test':
                if not entry.get('auto_tune', False):
                    required_fields = {
                        'warmup_batch', 'warmup_events', 'ramp_initial_batch', 'ramp_scale',
                        'ramp_events', 'search_events', 'search_tol_pct', 'steady_events',
                        'steady_eps_pct', 'steady_backoff_pct', 'grace_events', 'deadline_us',
                        'query_ratio'
                    }
                    missing = [f for f in required_fields if f not in entry]
                    if missing:
                        raise Exception(f"Missing fields for stress_test operation: {', '.join(missing)}")
            if entry['operation'] in {'replace'}:
                if 'tags_start' not in entry:
                    raise Exception('Start of indices to be replaced not specified in runbook')
                if 'tags_end' not in entry:
                    raise Exception('End of indices to be replaced not specified in runbook')
                if 'ids_start' not in entry:
                    raise Exception('Start of indices to replace not specified in runbook')
                if 'ids_end' not in entry:
                    raise Exception('End of indices to replace not specified in runbook')
                if entry['tags_start'] < 0 or entry ['tags_start'] >= max_pts:
                    raise Exception('Start of indices to be replaced out of range in runbook') 
                if entry['tags_end'] < 0 or entry ['tags_end'] > max_pts:
                    print(entry['tags_end'])
                    raise Exception('End of indices to be replaced out of range in runbook') 
                if entry['ids_start'] < 0 or entry ['ids_start'] >= max_pts:
                    raise Exception('Start of indices to replace out of range in runbook') 
                if entry['ids_end'] < 0 or entry ['ids_end'] > max_pts:
                    raise Exception('End of indices to replace out of range in runbook') 
            if entry['operation'] == 'auto_batch_delete':
                segment_size = entry.get('segment_size')
                segments = entry.get('segments')
                search_after = bool(entry.get('search_after_segment', False))
                wait_after = bool(entry.get('wait_after_segment', False))

                def _append_segment(seg_start: int, seg_end: int) -> None:
                    if seg_start < 0 or seg_end > max_pts or seg_start >= seg_end:
                        raise Exception('Invalid auto_batch_delete segment range')
                    seg_entry = {
                        'operation': 'batch_insert_delete',
                        'start': int(seg_start),
                        'end': int(seg_end),
                        'batchSize': entry['batchSize'],
                        'eventRate': entry['eventRate'],
                        'deletion_percentage': entry['deletion_percentage'],
                    }
                    run_list.append(seg_entry)
                    if search_after:
                        run_list.append({'operation': 'search'})
                    if wait_after:
                        run_list.append({'operation': 'waitPending'})

                if segments is not None:
                    if not isinstance(segments, list):
                        raise Exception('segments must be a list for auto_batch_delete')
                    for segment in segments:
                        if not isinstance(segment, dict) or 'start' not in segment or 'end' not in segment:
                            raise Exception('Invalid segment definition for auto_batch_delete')
                        _append_segment(int(segment['start']), int(segment['end']))
                else:
                    seg_start = entry['start']
                    seg_end_total = entry['end']
                    if segment_size is None:
                        segment_size_val = seg_end_total - seg_start
                    else:
                        segment_size_val = int(segment_size)
                    if segment_size_val <= 0:
                        raise Exception('segment_size must be positive for auto_batch_delete')
                    cursor = seg_start
                    while cursor < seg_end_total:
                        seg_end = cursor + segment_size_val
                        if seg_end > seg_end_total:
                            seg_end = seg_end_total
                        _append_segment(cursor, seg_end)
                        cursor = seg_end

                i += 1
                continue

            i += 1
            run_list.append(entry)
        
        max_pts = runbook.get('max_pts')
        if max_pts == None:
            raise Exception('max points not listed for dataset in runbook')
        maintenance_policy = runbook.get('maintenance_policy', {})
        return max_pts, RunbookConfig(run_list, maintenance_policy)

def get_gt_url(dataset_name, runbook_file):
    with open(runbook_file) as fd:
        runbook = yaml.safe_load(fd)[dataset_name]
        return runbook['gt_url']
