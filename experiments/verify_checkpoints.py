"""Read-only verification of every saved model; no fitting or downloading."""
import json
from pathlib import Path
import numpy as np
from checkpoints import ROOT, digest, load_checkpoint


def verify_all(directory=None):
    dest=Path(directory) if directory is not None else ROOT/'outputs/checkpoints'
    summary=json.loads((dest/'summary.json').read_text())
    expected={(d,m) for d in summary['config']['datasets'] for m in summary['config']['models']}
    assert expected=={(r['dataset'],r['model']) for r in summary['rows']}
    records=[]
    for row in summary['rows']:
        predictor=load_checkpoint(row['dataset'],row['model'],dest)
        assert digest(predictor.path/'manifest.json')==row['manifest_sha256']
        with np.load(dest/row['dataset']/'prepared.npz',allow_pickle=False) as data:
            train,val,test=[data[k] for k in ['train_indices','validation_indices','test_indices']]
            assert len(set(train)&set(val))==len(set(train)&set(test))==len(set(val)&set(test))==0
            assert sorted(np.concatenate([train,val,test]).tolist())==list(range(len(data['y'])))
            actual=predictor.predict(data['X'][test])
        expected_pred=np.load(predictor.path/'test_predictions.npy',allow_pickle=False)
        np.testing.assert_allclose(actual,expected_pred,rtol=1e-6,atol=1e-7)
        # A small independent coalition game exercises saved backgrounds and prediction API for all 24 models.
        instance,background=predictor.reference_game();base=predictor.predict(background).mean()
        grand=float(predictor.predict(instance[None,:])[0]-base)
        assert np.isfinite(grand)
        record={'dataset':row['dataset'],'model':row['model'],'test_rows':len(test),
                'reload_max_abs_error':float(np.max(np.abs(actual-expected_pred))),
                'empty_value':0.,'grand_coalition_value':grand}
        records.append(record);print('CHECKPOINT_VERIFIED '+json.dumps(record),flush=True)
    print('CHECKPOINT_VERIFICATION_SUMMARY '+json.dumps({'models':len(records),'status':'passed'}),flush=True)
    return records


if __name__=='__main__':verify_all()
