import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
pd.set_option('display.width',200)
d=pd.read_csv('data/train_test.csv',parse_dates=['date'])
d['lr']=np.log(d.posted_rate); d['ld']=np.log(d.distance)
d=d[(d.lr-d.groupby(pd.cut(d.ld,20),observed=True).lr.transform('median')).abs()<0.6]
X=pd.concat([d[['ld']],d[['ld']]**2,pd.get_dummies(d.equipment,drop_first=True)],axis=1).astype(float); X.columns=X.columns.astype(str)
d['res']=d.lr-LinearRegression().fit(X,d.lr).predict(X)
w=d.groupby(d.date.dt.to_period('M')).agg(res=('res','mean'),mi=('market_index','mean'),qs=('quote_signal','mean'))
print(w)
print('corr of residual w/ mi, qs (row level):',d[['res','market_index','quote_signal']].corr().iloc[0].round(3).to_dict())
d['m']=d.date.dt.month
for m,g in d.groupby('m'):
    print(m, 'corr res~mi %.3f  res~qs %.3f  slope res~mi %.3f  slope res~qs %.3f'%(g.res.corr(g.market_index),g.res.corr(g.quote_signal),np.polyfit(g.market_index.fillna(1),g.res,1)[0],np.polyfit(g.quote_signal,g.res,1)[0]))
# weekly daily-level relationship
dd=d.groupby('date').agg(res=('res','mean'),mi=('market_index','mean'),qs=('quote_signal','mean'))
print('daily corr',dd.corr().round(3))
for lag in [0,7,14,21,28]:
    print('lag',lag, round(dd.res.corr(dd.mi.shift(lag)),3))
print(dd.mi.rolling(7).mean().iloc[::20].round(3).tolist())
print(dd.res.rolling(7).mean().iloc[::20].round(3).tolist())
print('=====')
d['wk']=d.date.dt.to_period('W')
sl=d.groupby('wk').apply(lambda g: pd.Series({'slope_qs':np.polyfit(g.quote_signal,g.res,1)[0],'mi':g.market_index.mean(),'qs':g.quote_signal.mean(),'qs_sd':g.quote_signal.std()}),include_groups=False)
print(sl.round(3).T.to_string())
# does sign depend on equipment / distance?
for k,g in d.groupby('equipment'): print(k, np.polyfit(g.quote_signal,g.res,1)[0].round(3))
for k,g in d.groupby(pd.qcut(d.distance,4)): print(k, np.polyfit(g.quote_signal,g.res,1)[0].round(3))
# daily
ds=d.groupby('date').apply(lambda g: np.polyfit(g.quote_signal,g.res,1)[0],include_groups=False)
print(ds.iloc[:40].round(3).tolist())
print('#####')
d['mi_dev']=d.market_index-d.groupby('date').market_index.transform('mean')
d['res_dev']=d.res-d.groupby('date').res.transform('mean')
print('within-day corr res~mi_dev', d[['res_dev','mi_dev']].corr().iloc[0,1].round(3), ' slope', np.polyfit(d.mi_dev.dropna(), d.res_dev[d.mi_dev.notna()],1)[0].round(3))
dd=d.groupby('date').agg(res=('res','mean'),mi=('market_index','mean'))
hf=lambda s: s-s.rolling(15,center=True,min_periods=5).mean()
print('daily high-freq corr res~mi', hf(dd.res).corr(hf(dd.mi)).round(3), ' hf sd mi',hf(dd.mi).std().round(4),' hf sd res',hf(dd.res).std().round(4))
print('slope of daily res on mi (hf):', np.polyfit(hf(dd.mi).dropna(),hf(dd.res).dropna(),1)[0].round(3))
v=pd.read_csv('data/validation.csv',parse_dates=['date'])
vd=v.groupby('date').market_index.mean(); print('val daily mi hf sd', hf(vd).std().round(4), ' within-day sd', (v.market_index-v.groupby('date').market_index.transform('mean')).std().round(4))
print('train within-day sd', d.mi_dev.std().round(4))
