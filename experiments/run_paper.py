"""Run the table-specific paper experiments selected in committed configuration."""
import json
from paper_tables import ROOT,run_table,write_json,execution_fingerprint
from paper_california import run as california
from plot_figures import run as figures


def main():
    config=json.loads((ROOT/'configs/paper_reproduction.json').read_text())
    print('PAPER_REPRODUCTION_CONFIG '+json.dumps(config),flush=True)
    output=ROOT/config['output_directory']
    if config.get('operator_jobs'):
        from paper_measure import measure_one, metadata
        rows=[]
        for job in config['operator_jobs']:
            print('PAPER_CELL_START '+json.dumps(job),flush=True)
            row=measure_one(job)
            rows.append(row)
            print('PAPER_CELL_RESULT '+json.dumps(row),flush=True)
        result={'execution_fingerprint':execution_fingerprint(),
                'provenance':metadata(),'records':rows}
        write_json(output/'targeted-results.json',result)
        print('TARGETED_REMEASUREMENT_COMPLETE '+json.dumps(result),flush=True)
        if any(row['status']!='ok' for row in rows):
            raise RuntimeError('Targeted remeasurement includes failed cells; raw records preserved')
        return
    records=[]
    for table in config['tables']:
        data=run_table(table,output,table in config['refresh_tables'])
        records.append({'table':table,'content_sha256':data['content_sha256'],
                        'failed_cells':sum(r.get('status') in ['validation_failed','error'] for r in data['records']),
                        'oom_cells':sum(r.get('status')=='oom' for r in data['records'])})
    if config['california']:california(output/'california',config['refresh_california'])
    if config.get('figures',False):figures(output/'california',output/'figures')
    manifest={'execution_fingerprint':execution_fingerprint(),'tables':records,
              'california':config['california'],'figures':config.get('figures',False)}
    write_json(output/'manifest.json',manifest)
    print('PAPER_REPRODUCTION_COMPLETE '+json.dumps(manifest),flush=True)
    if any(row['failed_cells'] for row in records):
        raise RuntimeError('Remeasurement includes failed cells; inspect the preserved raw records')


if __name__=='__main__':main()
