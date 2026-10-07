import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from cv3 import d,dmi,lane_X,P,folds,build_features
def within_day_beta(trd):
    m=~trd.outlier.values&trd.market_index.notna().values; g=trd[m]
    # day fixed effects: demean y and mi within day AND within lane-type is unnecessary (mi noise independent of lane)
    yd=g.lr-g.groupby('date').lr.transform('mean'); xd=g.market_index-g.groupby('date').market_index.transform('mean')
    # control lane composition within day via log distance & equipment partial-out
    Z=np.c_[np.log(g.distance),np.log(g.distance)**2,pd.get_dummies(g.equipment).astype(float).values]
    Z=Z-pd.DataFrame(Z,index=g.index).groupby(g.date).transform('mean').values
    A=np.c_[xd,Z]; coef=np.linalg.lstsq(A,yd.values,rcond=None)[0]; return coef[0]
def fit_predict(trd,ted,use_mi=True,half_life=None):
    m=~trd.outlier.values; Xtr,Xte=lane_X(trd),lane_X(ted)
    mi_tr=trd.market_index.fillna(trd.date.map(dmi)).values; mi_te=ted.market_index.fillna(ted.date.map(dmi)).values
    beta=within_day_beta(trd) if use_mi else 0.0
    off_tr=beta*mi_tr; off_te=beta*mi_te
    w=np.ones(len(trd))
    if half_life: age=(trd.date.max()-trd.date).dt.days.values; w=0.5**(age/half_life)
    g=lgb.LGBMRegressor(**P).fit(Xtr[m],(trd.lr.values-off_tr)[m],sample_weight=w[m])
    return np.exp(g.predict(Xte)+off_te),beta
def score(trd,ted,**kw):
    p,b=fit_predict(trd,ted,**kw); y=ted.posted_rate.values; ok=~ted.outlier.values; e=np.abs(y-p)[ok]
    return e.mean(),(e/y[ok]).mean()*100,b
if __name__=='__main__':
    for name,kw in [('no mi, equal weights',dict(use_mi=False)),('no mi, HL=45d',dict(use_mi=False,half_life=45)),
                    ('mi(beta), equal weights',dict()),('mi(beta), HL=120d',dict(half_life=120)),('mi(beta), HL=60d',dict(half_life=60)),('mi(beta), HL=30d',dict(half_life=30)),('mi(beta), HL=15d',dict(half_life=15))]:
        res=[score(d[d.date<a],d[(d.date>=a)&(d.date<=b)],**kw) for a,b in folds]
        print(f'{name:26s}',' | '.join(f'{m:6.1f} ({p:4.2f}%)' for m,p,_ in res),'  beta~%.3f'%res[1][2])
