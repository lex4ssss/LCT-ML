import hashlib
import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve


source, episodes, run = map(Path,sys.argv[1:4])
selection = json.loads((run/'selection.json').read_text())
model = joblib.load(run/'model.joblib')
connection = duckdb.connect(config={'threads':1,'memory_limit':'1GB'})
connection.execute("CREATE TEMP TABLE all_rows AS SELECT * FROM read_parquet(?) WHERE split='validation'",[str(source)])
connection.execute('CREATE TEMP TABLE eligible AS SELECT *,row_number() OVER(ORDER BY channel_id,as_of) AS row_id FROM all_rows WHERE label IS NOT NULL')
columns = connection.execute('SELECT '+','.join(selection['features'])+',label FROM eligible ORDER BY row_id').fetchnumpy()
matrix = np.column_stack([np.ma.filled(columns[name].astype(float),np.nan) for name in selection['features']])
labels = columns['label']
scores = model.predict_proba(matrix)[:,1]
manual = np.log1p(np.nan_to_num(matrix,nan=24))
manual = (manual-model[2].mean_)/model[2].scale_
manual = 1/(1+np.exp(-(manual@model[-1].coef_[0]+model[-1].intercept_[0])))
np.testing.assert_allclose(scores,manual,rtol=1e-12,atol=1e-12)
predicted = scores>=selection['threshold']
tp, fp, fn, tn = [int(mask.sum()) for mask in [predicted&(labels==1),predicted&(labels==0),~predicted&(labels==1),~predicted&(labels==0)]]
assert abs(tp/(tp+fp)-selection['model']['precision'])<1e-12
assert abs(tp/(tp+fn)-selection['model']['recall'])<1e-12
connection.execute('CREATE TEMP TABLE scores AS SELECT unnest(?) AS row_id,unnest(?) AS score',[list(range(1,len(scores)+1)),scores.tolist()])
connection.execute("CREATE TEMP TABLE onsets AS SELECT channel_id,first_alarm_at,date_trunc('day',first_alarm_at-INTERVAL '1 microsecond') AS as_of FROM read_parquet(?) WHERE first_alarm_at>TIMESTAMP '2025-01-01' AND first_alarm_at<=TIMESTAMP '2026-01-01'",[str(episodes)])
coverage = connection.execute("SELECT CASE WHEN a.channel_id IS NULL THEN 'no_history_snapshot' WHEN a.label IS NULL THEN a.exclusion_reason ELSE 'eligible' END AS reason,count(*) FROM onsets e LEFT JOIN all_rows a USING(channel_id,as_of) GROUP BY reason ORDER BY reason").fetchall()
episode_hits = connection.execute('SELECT count(*),count(*) FILTER(WHERE s.score>=?) FROM onsets e JOIN eligible a USING(channel_id,as_of) JOIN scores s USING(row_id)',[selection['threshold']]).fetchone()
precision, recall, thresholds = precision_recall_curve(labels,scores)
bins = []
for lower,upper in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
    mask = (scores>=lower)&((scores<upper) if upper<1 else (scores<=upper))
    if mask.any():
        bins.append(dict(lower=float(lower),upper=float(upper),rows=int(mask.sum()),mean_score=float(scores[mask].mean()),observed_fraction=float(labels[mask].mean())))
constraints = {str(target):float(recall[:-1][precision[:-1]>=target].max(initial=0)) for target in [0.5,0.7]}
paths = [source,run/'model.joblib',Path(__file__).with_name('fit_baseline.py'),source.parent.parent.parent/'outputs/ml-dataset/dataset-config.json']
hashes = {str(path):hashlib.file_digest(path.open('rb'),'sha256').hexdigest() for path in paths}
report = dict(confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),alerts=int(predicted.sum()),false_alerts_per_1000_eligible_snapshots=1000*fp/len(labels),reliability_bins=bins,max_row_recall_at_precision=constraints,episode_coverage=dict(coverage),eligible_episodes=episode_hits[0],detected_eligible_episodes=episode_hits[1],coefficient=model[-1].coef_[0].tolist(),intercept=float(model[-1].intercept_[0]),sha256=hashes,flags=dict(validation=True,test_evaluation=False,training=False,offline_inference=True,calibration=False),manual_probability_check=True)
(run/'validation-diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
connection.close()
