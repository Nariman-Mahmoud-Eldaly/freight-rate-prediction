import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import HuberRegressor
from features import build_features
d=pd.read_csv('data/train_test.csv',parse_dates=['date']); v=pd.read_csv('data/validation.csv',parse_dates=['date'])
allx=pd.concat([d,v]); dmi=allx.groupby('date').market_index.median(); dqs=allx.groupby('date').quote_signal.mean()
d['lr']=np.log(d.posted_rate); T0=pd.Timestamp('2025-01-01')
X0=build_features(d,dmi).fillna(0)[['log_dist','equip','market_index','quote_signal']]
r=d.lr-HuberRegressor(max_iter=500).fit(X0,d.lr).predict(X0); d['outlier']=np.abs(r)>0.5
def tdays(df): return ((df.date-T0).dt.days/30.0).values   # months since Jan 1
def lane_X(df):
    f=build_features(df,dmi,use_market=False,use_quote=False); return f
P=dict(objective='huber',alpha=0.3,learning_rate=0.03,num_leaves=15,min_child_samples=60,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,n_estimators=700,verbose=-1)
def lin_design(df,use_trend):
    mi=df.market_index.fillna(df.date.map(dmi)).values
    cols=[mi]+([tdays(df)] if use_trend else [])
    return np.c_[tuple(cols)]
def fit_predict(trd,ted,use_trend=True,use_mi=True,huber_lin=True,qs_regime=False):
    m=~trd.outlier.values; Xtr,Xte=lane_X(trd),lane_X(ted)
    if qs_regime:
        for X,df in ((Xtr,trd),(Xte,ted)):
            dm=df.date.map(dqs).values; X['qs_day']=dm; X['qs_dev']=df.quote_signal.values-dm
    y=trd.lr.values
    # stage 1: joint linear fit gives beta for mi / trend, controlling for lane basics
    base=np.c_[np.log(trd.distance),np.log(trd.distance)**2,pd.get_dummies(trd.equipment).astype(float).values]
    Ztr=lin_design(trd,use_trend) if use_mi else (tdays(trd)[:,None] if use_trend else np.zeros((len(trd),0)))
    if not use_mi and not use_trend: off_tr=np.zeros(len(trd)); off_te=np.zeros(len(ted))
    else:
        Zte=lin_design(ted,use_trend) if use_mi else (tdays(ted)[:,None])
        lin=HuberRegressor(max_iter=1000).fit(np.c_[Ztr,base][m],y[m]); k=Ztr.shape[1]
        off_tr=Ztr@lin.coef_[:k]; off_te=Zte@lin.coef_[:k]
    g=lgb.LGBMRegressor(**P).fit(Xtr[m],(y-off_tr)[m])
    return np.exp(g.predict(Xte)+off_te)
def score(trd,ted,**kw):
    p=fit_predict(trd,ted,**kw); y=ted.posted_rate.values; ok=~ted.outlier.values; e=np.abs(y-p)[ok]
    return e.mean(),(e/y[ok]).mean()*100
folds=[('2025-07-01','2025-08-31'),('2025-09-01','2025-10-31'),('2025-08-01','2025-10-31'),('2025-10-01','2025-10-31')]
if __name__=='__main__':
    for name,kw in [('lane only (no mi, no trend)',dict(use_mi=False,use_trend=False)),
                    ('lane + trend',dict(use_mi=False,use_trend=True)),
                    ('lane + mi (linear)',dict(use_mi=True,use_trend=False)),
                    ('lane + mi + trend  <-- chosen',dict(use_mi=True,use_trend=True)),
                    ('  + qs regime feats',dict(use_mi=True,use_trend=True,qs_regime=True))]:
        res=[score(d[d.date<a],d[(d.date>=a)&(d.date<=b)],**kw) for a,b in folds]
        print(f'{name:32s}',' | '.join(f'{m:6.1f} ({p:4.2f}%)' for m,p in res))
