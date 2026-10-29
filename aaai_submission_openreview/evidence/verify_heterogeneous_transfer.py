"""Audit the supplied transfer archive; no training and no source mutations."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import io
import json
import statistics
import zipfile

BASE = Path(__file__).resolve().parent
archive = BASE / 'heterogeneous_transfer_results.zip'
with zipfile.ZipFile(archive) as bundle:
    state = json.loads(bundle.read('state.json'))
    protocol = state['protocol']
    assert protocol['partition'] == 'noniid-labeldir' and protocol['beta'] == .1
    assert protocol['rounds'] == 2000 and protocol['n_parties'] == 11
    assert protocol['momentum'] == .9 and not protocol['nesterov']
    assert not list(csv.DictReader(io.StringIO(bundle.read('tuning.csv').decode())))
    table = {r['method']: r for r in csv.DictReader(io.StringIO(bundle.read('table.csv').decode()))}
    assert len(table) == 10 and all(r['experiment'] == 'transfer' for r in table.values())
    records = defaultdict(list)
    metric_files = [n for n in bundle.namelist() if n.endswith('/metrics.json')]
    assert len(metric_files) == 50
    for name in metric_files:
        record = json.loads(bundle.read(name))
        config, history = record['config'], record['history']
        method, seed = config['algorithm'], config['seed']
        for key in ['dataset', 'model', 'rounds', 'n_parties', 'n_steps', 'batch_size',
                    'partition', 'beta', 'momentum', 'nesterov', 'lr_aux', 'weight_decay',
                    'cosine_schedule', 'freeze_bn_stats', 'ns_steps', 'lmo_dtype']:
            assert config[key] == protocol[key], (name, key)
        assert config['lr'] == protocol['homogeneous_lrs'][method]
        assert config['partition_seed'] == seed and config['split'] == 'full'
        assert history['steps'][-1] == 2000 and len(history['test_acc']) == len(history['steps'])
        records[method].append({'seed': seed, 'tail_mean': statistics.mean(history['test_acc'][-5:]),
                                'metrics_path': name, 'metrics_sha256': hashlib.sha256(bundle.read(name)).hexdigest()})
    results = {}
    for method, runs in records.items():
        runs.sort(key=lambda r: r['seed'])
        assert [r['seed'] for r in runs] == [0, 1, 2, 3, 4]
        values = [r['tail_mean'] for r in runs]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        assert abs(mean - float(table[method]['acc_mean'])) < 1e-10
        assert abs(sd - float(table[method]['acc_std'])) < 1e-10
        results[method] = {'mean': mean, 'sample_sd': sd, 'lr': protocol['homogeneous_lrs'][method], 'runs': runs}
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'source': 'Supplied code/results/rebuttal_results.zip; copied without modification during revision 2.',
              'protocol': protocol, 'verified_runs': 50, 'results': results,
              'scope': 'Fixed-IID-hyperparameter transfer under label skew. LR-retuned comparison is pending; no local model updates or partial participation.'}
    with (BASE / 'heterogeneous_transfer_verified.json').open('w', encoding='utf-8', newline='\n') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    for method, result in results.items():
        print(f"{method}: {result['mean']:.4f} +/- {result['sample_sd']:.4f}; 5 seeds")
    print('PASS: 50 complete transfer runs; configs, seed coverage and all table means/sample SDs.')
