from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from src.core import load_data,build_split,feature_matrix,Y_COL,LEAK_COL


def test_dataset_integrity(data_path):
    df,_,dupes,audit=load_data(data_path)
    assert audit['raw_rows']==1128
    assert len(df)==1128
    assert len(dupes)==11
    assert audit['conflicting_label_groups']==6
    assert len(set(df[Y_COL]))>100


def test_splits_no_canonical_leakage(data_path):
    df,*_=load_data(data_path)
    for mode in ['random','scaffold']:
        s=build_split(df,mode,42)
        canon=[set(df.iloc[s[k]].canonical_smiles) for k in ['train','val','test']]
        assert not canon[0]&canon[1] and not canon[1]&canon[2] and not canon[0]&canon[2]
        if mode=='scaffold':
            groups=[set(df.iloc[s[k]].scaffold) for k in ['train','val','test']]
            assert not groups[0]&groups[1] and not groups[1]&groups[2] and not groups[0]&groups[2]
        assert sum(len(s[k]) for k in ['train','val','test'])==1128


def test_features_no_external_predictions(data_path):
    df,*_=load_data(data_path)
    f,n=feature_matrix(df.iloc[:3])
    assert f['basic'].shape==(3,7)
    assert f['extended'].shape[1]==28
    assert f['combined'].shape[1]==1052
    assert not any(LEAK_COL in name for names in n.values() for name in names)
    assert np.isfinite(f['basic']).all()
