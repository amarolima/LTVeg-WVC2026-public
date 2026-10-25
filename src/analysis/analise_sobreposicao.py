#!/usr/bin/env python3
"""Mede sobreposicao geografica entre as imagens do dataset (extents de 162,5 m
em EPSG:3857) e o impacto da unica sobreposicao que cruza treino/teste.
Rodar da raiz: conda run -n research python src/analysis/analise_sobreposicao.py
Requer geo_meta.csv, produzido por analise_geografica.py.
"""
import pandas as pd, numpy as np, itertools
D=pd.read_csv('results/geo/geo_meta.csv')
# EPSG:4674 (~WGS84) -> EPSG:3857, que e o CRS em que o recorte foi feito
R=6378137.0
x=np.radians(D.lon.values)*R
y=R*np.log(np.tan(np.pi/4+np.radians(D.lat.values)/2))
S=650*0.25  # 162.5 m, lado do extent EM UNIDADES DE WEB MERCATOR
n=len(D)
dx=np.abs(x[:,None]-x[None,:]); dy=np.abs(y[:,None]-y[None,:])
ox=np.clip(S-dx,0,S); oy=np.clip(S-dy,0,S)
area=(ox*oy)/(S*S)            # fracao de sobreposicao de area
np.fill_diagonal(area,0)
iu=np.triu_indices(n,1)
a=area[iu]
print("pares totais:", len(a))
for t in [0.0,0.01,0.10,0.25,0.50]:
    print("  pares com sobreposicao > %4.0f%%: %d"%(t*100,(a>t).sum()))
print("\nimagens envolvidas em ALGUMA sobreposicao (>0):", (area.max(axis=1)>0).sum(), "de", n)
print("imagens com sobreposicao >10%%:", (area.max(axis=1)>0.10).sum())

# detalhar os pares que se sobrepoem
pi,pj=np.where(np.triu(area,1)>0)
print("\n--- pares sobrepostos ---")
for i,j in zip(pi,pj):
    r1,r2=D.iloc[i],D.iloc[j]
    d=np.hypot(x[i]-x[j],y[i]-y[j])
    print("  %.0f%% | %6dm | %s/%s(%s) x %s/%s(%s) | %s | %s"%(
      100*area[i,j], d, r1.split,r1.cls,r1.idx, r2.split,r2.cls,r2.idx,
      "MESMA LINHA" if r1.nome==r2.nome else "linhas dif",
      r1.nome[:45]))
# quantos cruzam fronteira de split
cross=sum(1 for i,j in zip(pi,pj) if D.iloc[i].split!=D.iloc[j].split)
print("\npares sobrepostos que cruzam splits diferentes:", cross, "de", len(pi))

# --- impacto da imagem vazada ---
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
z=np.load('results/nok_analysis/scores_test.npz',allow_pickle=True)
paths=np.array([str(p) for p in z['paths']]); y=z['y']; 
print("classe NOK =", z['y'][[i for i,p in enumerate(paths) if 'NOK' in p]][:3], "(1=NOK)")
mask=np.array(['172788' not in p for p in paths])
print("imagem vazada encontrada:", (~mask).sum(), [p for p in paths if '172788' in p])
for name in ['resnet','yolo']:
    s=z[name]
    for tag,m in [("todas as 69 imgs",np.ones(len(y),bool)),("sem a vazada (68)",mask)]:
        pred=(s[m]>0.5).astype(int)
        print("%-8s %-20s macroF1 %.1f%%  ROC-AUC %.3f  AP %.3f  NOK recall %.1f%% (n_NOK=%d)"%(
            name,tag,100*f1_score(y[m],pred,average='macro'),roc_auc_score(y[m],s[m]),
            average_precision_score(y[m],s[m]),
            100*f1_score(y[m],pred,pos_label=1,average='binary',zero_division=0) if False else 100*((pred[y[m]==1]==1).mean()),
            (y[m]==1).sum()))
    # o modelo acertou a imagem vazada?
    i=np.where(~mask)[0][0]
    print("         -> na imagem vazada: score=%.3f, rotulo=%d, %s"%(s[i],y[i],"ACERTOU" if (s[i]>0.5)==(y[i]==1) else "ERROU"))
