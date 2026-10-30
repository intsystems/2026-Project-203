"""Recompute validation selections and all final rows from the frozen archive."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile


BASE = Path(__file__).resolve().parent
ARCHIVE = BASE / 'heterogeneous_tuning_results.zip'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit():
    with zipfile.ZipFile(ARCHIVE) as bundle:
        state = json.loads(bundle.read('state.json'))
        protocol, jobs = state['protocol'], state['jobs']
        assert protocol['dataset'] == 'cifar10' and protocol['model'] == 'cnn2'
        assert protocol['partition'] == 'noniid-labeldir' and protocol['beta'] == .1
        assert protocol['rounds'] == 2000 and protocol['n_parties'] == 11
        assert protocol['n_steps'] == 3 and protocol['batch_size'] == 64
        assert protocol['momentum'] == .9 and not protocol['nesterov']
        assert protocol['seeds'] == [0, 1, 2, 3, 4] and protocol['tune_seeds'] == [0]
        assert state['tuning_protocol']['grid'] == '1-2-5'
        assert state['tuning_protocol']['initial_points'] == 5
        assert state['tuning_protocol']['extension_points'] == 2
        assert state['tuning_protocol']['extension_rounds'] == 3
        methods = protocol['methods']
        assert len(set(methods)) == 10
        metric_names = {name for name in bundle.namelist() if name.endswith('/metrics.json')}
        assert len(jobs) == len(metric_names) == 167
        assert {row['metrics'] for row in jobs.values()} == metric_names
        rows = list(csv.DictReader(io.StringIO(bundle.read('table.csv').decode('utf-8'))))
        table = {(row['experiment'], row['method']): row for row in rows}
        assert len(table) == 20
        trials = list(csv.DictReader(io.StringIO(bundle.read('tuning.csv').decode('utf-8'))))
        csv_runs = list(csv.DictReader(io.StringIO(bundle.read('runs.csv').decode('utf-8'))))
        assert len(trials) == 72 and len(csv_runs) == 167
        trial_paths = {row['metrics'] for row in trials}
        assert len(trial_paths) == 72
        assert {row['metrics'] for row in csv_runs} == metric_names
        verified = {}
        split_counts = Counter()
        for key, job in jobs.items():
            assert job['status'] == 'ok', key
            raw = bundle.read(job['metrics'])
            record = json.loads(raw)
            config, history = record['config'], record['history']
            for field in ('dataset', 'model', 'rounds', 'n_parties', 'n_steps', 'batch_size',
                          'partition', 'beta', 'nesterov', 'lr_aux', 'lr_scaling', 'weight_decay',
                          'cosine_schedule', 'freeze_bn_stats', 'ns_steps', 'lmo_dtype',
                          'val_seed', 'mv_ties', 'uplink_zeros', 'last_k'):
                assert config[field] == protocol[field], (key, field)
            assert config['algorithm'] == job['method']
            assert config['lr'] == job['lr'] and config['momentum'] == job['momentum'] == .9
            assert config['seed'] == config['partition_seed'] == job['seed'] == job['partition_seed']
            assert config['split'] == job['split']
            assert config['n_matrix_params'] == 762560 and config['n_aux_params'] == 2146
            assert history['steps'][-1] == 2000
            assert all(a < b for a, b in zip(history['steps'], history['steps'][1:]))
            split = job['split']
            split_counts[split] += 1
            metric = 'val_acc' if split == 'tune' else 'test_acc'
            other_metric = 'test_acc' if split == 'tune' else 'val_acc'
            assert metric in history and other_metric not in history, (key, 'evaluation split')
            assert len(history[metric]) == len(history['steps']) >= 5
            assert all(math.isfinite(value) for value in history[metric])
            tail = statistics.mean(history[metric][-5:])
            assert math.isclose(tail, job[metric], abs_tol=1e-12)
            assert (job['metrics'] in trial_paths) == (split == 'tune')
            verified[key] = dict(method=job['method'], seed=job['seed'], split=split,
                                 lr=job['lr'], score=tail, metrics_path=job['metrics'],
                                 metrics_sha256=digest(raw))
        assert split_counts == Counter(tune=72, full=95)
        selections, results = {}, {'transfer': {}, 'tuned': {}}
        for method in methods:
            search = state['searches'][method]
            grid = search['grid']
            assert grid == sorted(set(grid)) and all(value > 0 for value in grid)
            assert search['complete'] and not search['boundary']
            if method != 'muonsign':
                assert search['extensions'] <= 3
            else:
                assert search['extensions'] == 4
                assert search['refinement']['complete']
                assert search['refinement']['previous_lr'] == .00005
            candidate_records = [row for row in verified.values() if row['method'] == method and row['split'] == 'tune']
            assert len(candidate_records) == len(grid)
            assert {row['lr'] for row in candidate_records} == set(grid)
            assert all(row['seed'] == 0 for row in candidate_records)
            # Independently reproduce the documented validation-only rule and tie break.
            selected = min(candidate_records, key=lambda row: (
                -row['score'], abs(math.log(row['lr'] / protocol['homogeneous_lrs'][method])), row['lr']))
            assert min(grid) < selected['lr'] < max(grid)
            selections[method] = dict(lr=selected['lr'], validation_accuracy=selected['score'],
                                       candidates=len(grid), grid=grid, interior=True,
                                       boundary_extensions=search['extensions'])
            for experiment in ('transfer', 'tuned'):
                lr = protocol['homogeneous_lrs'][method] if experiment == 'transfer' else selected['lr']
                final = sorted((row for row in verified.values() if row['method'] == method and
                                row['split'] == 'full' and row['lr'] == lr), key=lambda row: row['seed'])
                assert [row['seed'] for row in final] == [0, 1, 2, 3, 4]
                values = [row['score'] for row in final]
                mean, sd = statistics.mean(values), statistics.stdev(values)
                row = table[experiment, method]
                assert row['complete'] == 'True' and int(row['seeds']) == int(row['expected_seeds']) == 5
                assert float(row['lr']) == lr and float(row['momentum']) == .9
                assert math.isclose(float(row['acc_mean']), mean, abs_tol=1e-12)
                assert math.isclose(float(row['acc_std']), sd, abs_tol=1e-12)
                results[experiment][method] = dict(lr=lr, mean=mean, sample_sd=sd, runs=final)
        assert {method for method, row in results['tuned'].items() if row['mean'] > 80} == {'ef21muonsign', 'ef21muonusign'}
        # The five full-precision Muon runs are shared by the two experiment rows.
        shared = set(row['metrics_path'] for row in results['transfer']['muon']['runs'])
        assert shared == set(row['metrics_path'] for row in results['tuned']['muon']['runs'])
        old_report = json.loads((BASE / 'heterogeneous_transfer_verified.json').read_text(encoding='utf-8'))
        for method, row in old_report['results'].items():
            current = results['transfer'][method]
            assert current['mean'] == row['mean'] and current['sample_sd'] == row['sample_sd']
            assert [record['metrics_sha256'] for record in current['runs']] == [record['metrics_sha256'] for record in row['runs']]
        report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), archive=ARCHIVE.name,
                      archive_sha256=digest(ARCHIVE.read_bytes()), source='Supplied code/results/rebuttal_results4.zip',
                      protocol=protocol, search_protocol=state['tuning_protocol'],
                      unique_runs=167, validation_runs=72, unique_final_runs=95,
                      final_table_rows=20, final_seed_memberships=100, shared_final_runs=5,
                      all_runs_reached_2000=True, validation_only_selection_verified=True,
                      all_selected_rates_interior=True, original_50_transfer_runs_unchanged=True,
                      muonsign_extra_validation_refinement=dict(rates=[.00001, .00002],
                                                               selected_lr_unchanged=.00005, extra_runs=2),
                      selections=selections, results=results,
                      scope='LR-only validation selection at momentum 0.9; seed 0 tuning and five final seeds. No local parameter updates or partial participation. Not a joint LR/momentum optimum.')
        with (BASE / 'heterogeneous_tuning_verified.json').open('w', encoding='utf-8', newline='\n') as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write('\n')
        for method, row in sorted(results['tuned'].items(), key=lambda item: -item[1]['mean']):
            print(f"{method}: LR {row['lr']:g}; {row['mean']:.4f} +/- {row['sample_sd']:.4f}; 5 seeds")
        print('PASS: 167 completed runs; 72 validation trials; 20 five-seed rows; all 10 LR winners interior; original transfer results unchanged.')


if __name__ == '__main__':
    audit()
