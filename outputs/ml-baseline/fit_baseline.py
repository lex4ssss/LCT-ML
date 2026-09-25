import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


source, output = map(Path, sys.argv[1:3])
if output.exists():
    raise FileExistsError(str(output))
features = ['event_count_24h','alarm_count_24h','last_event_age_hours','last_alarm_age_hours_24h']
connection = duckdb.connect(config={'threads':1,'memory_limit':'1GB'})


def read_split(split):
    columns = connection.execute('SELECT '+','.join(features)+',label FROM read_parquet(?) WHERE split=? AND label IS NOT NULL ORDER BY channel_id,as_of',[str(source),split]).fetchnumpy()
    matrix = np.column_stack([np.ma.filled(columns[name].astype(float),np.nan) for name in features])
    labels = columns['label']
    if set(np.unique(labels)) != {0,1} or np.any(matrix[~np.isnan(matrix)]<0):
        raise ValueError('Invalid training or validation data')
    return matrix, labels


def metrics(labels, scores, threshold):
    precision, recall, f1, _ = precision_recall_fscore_support(labels,scores>=threshold,average='binary',zero_division=0)
    return dict(precision=float(precision),recall=float(recall),f1=float(f1),average_precision=float(average_precision_score(labels,scores)),brier=float(brier_score_loss(labels,scores)))


train_x, train_y = read_split('train')
valid_x, valid_y = read_split('validation')
model = make_pipeline(SimpleImputer(strategy='constant',fill_value=24),FunctionTransformer(np.log1p),StandardScaler(),LogisticRegression(C=1,max_iter=500,tol=1e-6,solver='lbfgs'))
model.fit(train_x,train_y)
scores = model.predict_proba(valid_x)[:,1]
precision, recall, thresholds = precision_recall_curve(valid_y,scores)
f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
threshold = float(thresholds[np.flatnonzero(f1==f1.max())[-1]])
report = dict(features=features,threshold=threshold,threshold_selection='maximum_validation_row_f1_highest_threshold_tie',train_rows=len(train_y),validation_rows=len(valid_y),train_prevalence=float(train_y.mean()),validation_prevalence=float(valid_y.mean()),model=metrics(valid_y,scores,threshold),persistence=metrics(valid_y,(valid_x[:,1]>0).astype(float),0.5),constant=metrics(valid_y,np.full(len(valid_y),train_y.mean()),0.5),iterations=model[-1].n_iter_.tolist(),sklearn_version=sklearn.__version__,training=True,validation=True,test_evaluation=False,calibration=False,class_weight=None)
output.mkdir(parents=True)
joblib.dump(model,output/'model.joblib')
(output/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
connection.close()
