"""Generate, save, independently spot-check and reload every declared paper game."""
import json
import numpy as np
from unittest.mock import patch
from model_games import settings,checkpoint_game
from checkpoints import load_checkpoint,ROOT
from paper_tables import GAME_CONFIG,write_json
from paper_measure import paper_input

# Tree predictors evaluate one row at a time in exact arithmetic, so a coalition
# value recomputed from a differently shaped batch matches bitwise. The MLP runs
# float32 GEMMs whose blocking depends on the batch shape, so an independent
# recomputation only agrees to float32 precision.
SPOT_TOLERANCE={'dt':1e-12,'lgbm':1e-12,'xgb':1e-12,'mlp':1e-6}


def spot_check(model,values,instance,background,d):
    """Rebuild selected coalition values one batch at a time, outside the generator."""
    base=float(model.predict(background).mean());max_error=0.
    for mask in sorted({0,1,3,1<<(d-1),(1<<d)//3,(1<<d)-1}):
        hybrid=background.copy()
        for bit in range(d):
            if mask&(1<<bit):hybrid[:,bit]=instance[bit]
        max_error=max(max_error,abs(values[mask]-(float(model.predict(hybrid).mean())-base)))
    return max_error


def main():
    config=settings();rows=[];seen={};families={}
    for experiment,dimensions in sorted(GAME_CONFIG['dimensions_by_experiment'].items()):
        family=config['model_by_experiment'][experiment]
        for d in dimensions:
            dataset=config['dataset_by_dimension'][str(d)]
            print('MODEL_GAME_START '+json.dumps({'experiment':experiment,'dataset':dataset,'model':family,'d':d}),flush=True)
            values,info=paper_input(d,experiment)
            key=(dataset,family)
            if key in seen:
                assert info['sha256']==seen[key],(key,'shared game differs between experiments')
                print('MODEL_GAME_SHARED '+json.dumps({'experiment':experiment,'dataset':dataset,'model':family,'d':d,'input_sha256':info['sha256']}),flush=True)
                continue
            model=load_checkpoint(dataset,family);instance,background=model.reference_game()
            max_error=spot_check(model,values,instance,background,d)
            tolerance=SPOT_TOLERANCE[family]
            assert max_error<tolerance,(dataset,family,max_error,tolerance)
            assert values.shape==(1<<d,) and values[0]==0 and np.isfinite(values).all()
            with patch('model_games.generate_values',side_effect=AssertionError('cache hit regenerated values')):
                cached,_=checkpoint_game(dataset,family)
            np.testing.assert_array_equal(values,cached)
            seen[key]=info['sha256'];families[family]=families.get(family,0)+1
            row={'dataset':dataset,'d':d,'model':family,'experiments':[experiment],'n_values':len(values),
                 'input_sha256':info['sha256'],'max_prediction_spot_error':max_error,'spot_tolerance':tolerance,
                 'reload_equal':True,'input':info}
            rows.append(row);print('MODEL_GAME_VERIFIED '+json.dumps(row),flush=True)
    for experiment,dimensions in sorted(GAME_CONFIG['dimensions_by_experiment'].items()):
        family=config['model_by_experiment'][experiment]
        for d in dimensions:
            row=next(r for r in rows if (r['dataset'],r['model'])==(config['dataset_by_dimension'][str(d)],family))
            if experiment not in row['experiments']:row['experiments'].append(experiment)
    assert set(families)=={'dt','lgbm','xgb','mlp'},families
    write_json(ROOT/'outputs/games/summary.json',
               {'rows':rows,'games':len(rows),'families':families,
                'models_by_experiment':config['model_by_experiment'],'all_verified':True})
    print('MODEL_GAMES_COMPLETE '+json.dumps({'games':len(rows),'families':families,
          'coalition_values':sum(r['n_values'] for r in rows)}),flush=True)


if __name__=='__main__':main()
