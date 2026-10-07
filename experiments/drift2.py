import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
d=pd.read_csv('data/train_test.csv',parse_dates=['date'])
d['lr']=np.log(d.posted_rate); d['ld']=np.log(d.distance)
d=d[(d.lr-d.groupby(pd.cut(d.ld,20),observed=True).lr.transform('median')).abs()<0.6].copy()
X=pd.concat([d[['ld']],d[['ld']]**2,pd.get_dummies(d.equipment,drop_first=True)],axis=1).astype(float); X.columns=X.columns.astype(str)
d['res']=d.lr-LinearRegression().fit(X,d.lr).predict(X)
d['mi_dev']=d.market_index-d.groupby('date').market_index.transform('mean')
d['res_dev']=d.res-d.groupby('date').res.transform('mean')
d['m']=d.date.dt.month
ok=d.mi_dev.notna()
print('within-day slope of res on mi_dev by month (causal-ish):')
print(d[ok].groupby('m').apply(lambda g: pd.Series({'slope':np.polyfit(g.mi_dev,g.res_dev,1)[0],'corr':g.mi_dev.corr(g.res_dev)}),include_groups=False).round(3).T.to_string())
dd=d.groupby('date').agg(res=('res','mean'),mi=('market_index','mean')); dd['m']=dd.index.month
print('daily-level slope res~mi by month:')
print(dd.groupby('m').apply(lambda g: np.polyfit(g.mi,g.res,1)[0],include_groups=False).round(3).to_string())
# is relationship really lagged/delayed? correlate res with mi shifted by k days, by half-year
for lag in [-14,-7,0,7,14]:
    print('lag',lag,'corr res vs mi(t-lag):',round(dd.res.corr(dd.mi.shift(lag)),3), ' Jan-Jun',round(dd.res[:'2025-06-30'].corr(dd.mi.shift(lag)[:'2025-06-30']),3),' Jul-Oct',round(dd.res['2025-07-01':].corr(dd.mi.shift(lag)['2025-07-01':]),3))
print('######')
dd['s']=dd.res-0.13*dd.mi
w=dd.s.resample('10D').mean()
print(w.round(3).to_string())
import numpy as np
print('weekday pattern of s:', (dd.s-dd.s.rolling(15,center=True,min_periods=5).mean()).groupby(dd.index.dayofweek).mean().round(4).to_dict())
