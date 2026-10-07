import numpy as np, pandas as pd
from cv4 import d,folds,fit_predict
def ev(ted,p):
    y=ted.posted_rate.values; ok=~ted.outlier.values; e=np.abs(y-p)[ok]; return e.mean(),(e/y[ok]).mean()*100
rows={k:[] for k in ['A no-mi','B mi HL30','C blend A+B (log avg)','D no-mi HL45 + mi HL30 + no-mi eq (3-way)']}
for a,b in folds:
    tr,te=d[d.date<a],d[(d.date>=a)&(d.date<=b)]
    pa,_=fit_predict(tr,te,use_mi=False); pb,_=fit_predict(tr,te,half_life=30); pc,_=fit_predict(tr,te,use_mi=False,half_life=45)
    rows['A no-mi'].append(ev(te,pa)); rows['B mi HL30'].append(ev(te,pb))
    rows['C blend A+B (log avg)'].append(ev(te,np.sqrt(pa*pb)))
    rows['D no-mi HL45 + mi HL30 + no-mi eq (3-way)'].append(ev(te,(pa*pb*pc)**(1/3)))
for k,v in rows.items(): print(f'{k:44s}',' | '.join(f'{m:6.1f} ({p:4.2f}%)' for m,p in v),'  mean MAPE %.2f'%np.mean([p for _,p in v]))
