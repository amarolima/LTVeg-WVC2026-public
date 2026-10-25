#!/usr/bin/env python3
"""Reconstroi linha de transmissao e coordenada de cada imagem do dataset a partir
do indice no nome do arquivo, e mede o vazamento geografico entre os splits.
Rodar da raiz do projeto: conda run -n research python src/analysis/analise_geografica.py
"""
import pandas as pd, glob, os, numpy as np
d=pd.read_excel('data/metadata/coordenadas_extraidas.xlsx')
rows=[]
for split in ['train','val','test']:
    for cls in ['OK','NOK']:
        for p in glob.glob(f'data/dataset/{split}/{cls}/*.png'):
            idx=int(os.path.basename(p).split('_')[0]); r=d.iloc[idx]
            rows.append(dict(split=split,cls=cls,idx=idx,nome=r['Nome'],tensao=r['Tensao'],
                             lon=r['X_coord_pre'],lat=r['Y_coord_pre']))
D=pd.DataFrame(rows)
print("total imagens:", len(D)); print(D.groupby(['split','cls']).size())
print("\nlinhas de transmissao distintas:", D.nome.nunique())
tr=set(D[D.split=='train'].nome); te=D[D.split=='test']; va=D[D.split=='val']
print("test com linha vista no train: %d/%d (%.0f%%)"%(te.nome.isin(tr).sum(),len(te),100*te.nome.isin(tr).mean()))
print("val  com linha vista no train: %d/%d (%.0f%%)"%(va.nome.isin(tr).sum(),len(va),100*va.nome.isin(tr).mean()))
def hav(lat1,lon1,lat2,lon2):
    R=6371.0;p=np.pi/180
    a=np.sin((lat2-lat1)*p/2)**2+np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
T=D[D.split=='train']
dists=np.array([hav(r.lat,r.lon,T.lat.values,T.lon.values).min() for _,r in te.iterrows()])
print("\ndist ao vizinho de treino (km): min %.3f p25 %.1f mediana %.1f p75 %.1f max %.0f"%(
 dists.min(),np.percentile(dists,25),np.median(dists),np.percentile(dists,75),dists.max()))
for t in [0.2,1,5,20,50]: print("  <%5.1f km: %d/%d"%(t,(dists<t).sum(),len(dists)))
os.makedirs('results/geo',exist_ok=True)
D.to_csv('results/geo/geo_meta.csv',index=False)
