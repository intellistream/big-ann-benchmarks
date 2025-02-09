import yaml

def load_runbook(dataset_name, max_pts, runbook_file):
    with open(runbook_file) as fd:
        runbook = yaml.safe_load(fd)[dataset_name]
        
    write_ratio = runbook.get('write_ratio', 0.5)  
    batch_size = runbook.get('batch_size', 100)    

    run_list = []
    i = 1
    while str(i) in runbook:  
        entry = runbook[str(i)]
        
        if entry['operation'] not in {'insert_and_search'}:
            raise Exception('Undefined runbook operation')

        if 'start' not in entry or 'end' not in entry:
            raise Exception(f'Start/End missing in runbook at entry {i}')
        
        if entry['start'] < 0 or entry['start'] >= max_pts:
            raise Exception(f'Start out of range at entry {i}')
        if entry['end'] < 0 or entry['end'] > max_pts:
            raise Exception(f'End out of range at entry {i}')

        run_list.append(entry)
        i += 1

    gt_url = runbook.get('gt_url', "none")

    return write_ratio, batch_size, gt_url, run_list

def get_gt_url(dataset_name, runbook_file):
    with open(runbook_file) as fd:
        runbook = yaml.safe_load(fd)[dataset_name]
        return runbook.get('gt_url', "none")  
