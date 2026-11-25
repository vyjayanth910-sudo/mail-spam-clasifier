"""Reproducible email + SMS model training. Run from the project root."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from textclean import normalize
from train_v3 import build

ROOT = Path(__file__).resolve().parents[1]

def load_sms():
    sms = pd.read_csv(ROOT/'data'/'SMSSpamCollection', sep='\t', header=None, names=['label','text'], encoding='latin1')
    return pd.DataFrame({'text':sms.text, 'label':sms.label.map({'ham':0,'spam':1}), 'src':'sms'})

def data():
    # Legacy loaders expect a working directory of the project root.
    import os
    os.chdir(ROOT)
    email = build('b')
    sms = load_sms()
    email = pd.concat([g.sample(n=min(len(g), 2500), random_state=42) for _,g in email.groupby(['src','label'])],ignore_index=True)
    d = pd.concat([email,sms], ignore_index=True)
    d = d.dropna(subset=['text','label'])
    d['text'] = d.text.astype(str).str.strip()
    d = d[d.text.str.len() >= 4]
    d['key'] = d.text.map(normalize)
    d = d[d.key.str.len() >= 4].drop_duplicates('key').drop(columns='key')
    return d

def model():
    return Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=normalize, ngram_range=(1,1), min_df=3, max_features=30000, sublinear_tf=True, strip_accents='unicode')),
        ('classifier', LogisticRegression(max_iter=250, class_weight='balanced', solver='liblinear'))
    ])

def metrics(y, scores, threshold):
    pred = (scores>=threshold).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp), 'ham_specificity':round(tn/(tn+fp),4), 'scam_recall':round(tp/(tp+fn),4), 'precision':round(tp/(tp+fp),4) if tp+fp else 0}

if __name__=='__main__':
    d = data()
    tr, temp = train_test_split(d,test_size=.30,random_state=42,stratify=d.src+'_'+d.label.astype(str))
    val, test = train_test_split(temp,test_size=.5,random_state=43,stratify=temp.src+'_'+temp.label.astype(str))
    m = model().fit(tr.text,tr.label)
    scores=m.predict_proba(val.text)[:,1]
    # Set threshold using validation only; prioritize avoiding false positives while preserving recall.
    candidates=np.arange(.50,.991,.01)
    feasible=[(t,metrics(val.label,scores,t)) for t in candidates]
    eligible=[(t,r) for t,r in feasible if r['ham_specificity']>=.985 and r['scam_recall']>=.90]
    threshold=float(min(eligible,key=lambda x:x[0])[0] if eligible else max(feasible,key=lambda x:x[1]['ham_specificity']+x[1]['scam_recall'])[0])
    report={'train_rows':len(tr),'validation_rows':len(val),'test_rows':len(test),'threshold':round(threshold,3), 'validation':metrics(val.label,scores,threshold)}
    test_scores=m.predict_proba(test.text)[:,1]
    report['test']=metrics(test.label,test_scores,threshold)
    report['test_by_source']={s:metrics(g.label,m.predict_proba(g.text)[:,1],threshold) for s,g in test.groupby('src') if g.label.nunique()==2}
    indian=pd.read_csv(ROOT/'data'/'indian_test.csv')
    indian['score']=m.predict_proba(indian.text)[:,1]
    indian['prediction']=np.where(indian.score>=threshold,'scam','ham')
    report['indian_diagnostic']=[{'text':r.text,'actual':r.label,'predicted':r.prediction,'score':round(float(r.score),3)} for r in indian.itertuples()]
    # Fit final model on training + validation, leaving test untouched.
    final=model().fit(pd.concat([tr.text,val.text]),pd.concat([tr.label,val.label]))
    joblib.dump({'model':final,'threshold':threshold,'version':'final_v1'},ROOT/'models'/'final_v1.joblib')
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports'/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
