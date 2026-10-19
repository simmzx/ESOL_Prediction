import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.core import load_data,build_split,feature_matrix,dump_json,Y_COL


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--split',choices=['random','scaffold'],default='random')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--fp-bits',type=int,default=1024)
    args=ap.parse_args()
    p=Path(args.output);p.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    df,dropped,duplicates,audit=load_data(args.data)
    audit['source_sha256']=sha
    audit['original_csv_path']=str(Path(args.data).resolve())
    split=build_split(df,args.split,args.seed)
    split['source_sha256']=sha
    df.to_csv(p/'clean_data.csv',index=False)
    dropped.to_csv(p/'excluded_invalid.csv',index=False)
    duplicates.to_csv(p/'duplicate_structures.csv',index=False)
    dump_json(p/'audit.json',audit)
    dump_json(p/'split.json',split)
    memberships=np.empty(len(df),dtype=object)
    for k in ('train','val','test'):
        memberships[split[k]]=k
    info=df[['Compound ID','smiles','canonical_smiles','scaffold',Y_COL]].copy()
    info['subset']=memberships
    info.to_csv(p/'split_membership.csv',index=False)
    f,names=feature_matrix(df,args.fp_bits)
    np.savez_compressed(p/'features.npz',**f)
    dump_json(p/'feature_names.json',names)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(df[Y_COL],bins=35,edgecolor='white')
    axes[0].set(xlabel='Measured logS (log10 mol/L)',ylabel='Count',title='ESOL target distribution')
    for k in ['train','val','test']:
        axes[1].hist(df.iloc[split[k]][Y_COL],bins=30,alpha=0.5,label=f'{k}: {len(split[k])}')
    axes[1].set(xlabel='Measured logS (log10 mol/L)',ylabel='Count',title=f'{args.split.capitalize()} split')
    axes[1].legend();fig.tight_layout();fig.savefig(p/'distribution.png',dpi=180);plt.close(fig)
    print('[PREPARE] audit',audit)
    print('[PREPARE] counts',split['counts'],'feature dimensions',{k:list(v.shape) for k,v in f.items()})

if __name__=='__main__':main()
