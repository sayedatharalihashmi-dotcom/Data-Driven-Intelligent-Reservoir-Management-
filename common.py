import os, numpy as np, pandas as pd
V='Manual Daily River Water Discharge (m3/sec)'
def load():
    df=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'river_discharge_manual_daily_chhattisgarh_sw_cg_2001_2025.csv'))
    d=df[df.Station.str.contains('Arang')].copy()
    d['date']=pd.to_datetime(d['Data Acquisition Time'],format='%d-%m-%Y %H:%M').dt.normalize()
    d=d.sort_values('date').reset_index(drop=True)
    return d[['date',V]].rename(columns={V:'q'})
def metrics(o,p):
    o=np.asarray(o,float);p=np.asarray(p,float)
    rmse=np.sqrt(np.mean((o-p)**2)); mae=np.mean(abs(o-p))
    nse=1-((o-p)**2).sum()/((o-o.mean())**2).sum()
    r=np.corrcoef(o,p)[0,1] if p.std()>0 else 0
    kge=1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)
    pb=100*(p-o).sum()/o.sum()
    return dict(RMSE=rmse,MAE=mae,NSE=nse,KGE=kge,PBIAS=pb)
