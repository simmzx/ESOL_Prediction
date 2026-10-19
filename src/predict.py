"""Inference on new unlabeled SMILES with a saved trained ML or RNN checkpoint.
Examples:
python -m src.predict --ml-model results/.../ml/selected_model.joblib --smiles 'CCO' 'c1ccccc1'
python -m src.predict --rnn-checkpoint results/.../rnn/teacher_bilstm/seed_42/checkpoint.pt --smiles 'CCO'
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from src.core import feature_matrix


def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--ml-model')
    g.add_argument('--rnn-checkpoint')
    inputs=ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--smiles',nargs='+')
    inputs.add_argument('--csv',help='Input CSV with a smiles column')
    ap.add_argument('--output',default='predictions_new.csv')
    args=ap.parse_args()
    df=pd.DataFrame({'smiles':args.smiles}) if args.smiles else pd.read_csv(args.csv)
    if 'smiles' not in df:raise ValueError('Input must have smiles column')
    canonical=[]
    for s in df.smiles:
        mol=Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
        if mol is None:raise ValueError(f'Invalid SMILES: {s!r}')
        canonical.append(Chem.MolToSmiles(mol,isomericSmiles=True))
    df['canonical_smiles']=canonical
    if args.ml_model:
        pack=joblib.load(args.ml_model)
        X,_=feature_matrix(df)
        pred=pack['model'].predict(X[pack['representation']])
        model_name=pack['name']
    else:
        import torch
        from src.train_rnn import TeacherBiLSTM,BigruAttention
        state=torch.load(args.rnn_checkpoint,map_location='cpu',weights_only=False)
        vocab=state['vocab'];mx=state['max_length']
        arr=np.zeros((len(df),mx),dtype=np.int64);lengths=np.empty(len(df),dtype=np.int64)
        for i,s in enumerate(df.smiles.astype(str)):
            seq=[vocab.get(c,1) for c in s[:mx]]
            arr[i,:len(seq)]=seq;lengths[i]=max(len(seq),1)
        model=TeacherBiLSTM(len(vocab)+2) if state['architecture']=='teacher_bilstm' else BigruAttention(len(vocab)+2)
        model.load_state_dict(state['state_dict']);model.eval()
        with torch.no_grad():
            pred=model(torch.tensor(arr),torch.tensor(lengths)).numpy()*state['y_std']+state['y_mean']
        model_name=state['architecture']
    df['predicted_logS']=np.asarray(pred).reshape(-1)
    df.to_csv(args.output,index=False)
    print(f'[PREDICT] {model_name} -> {args.output}')
    print(df[['smiles','predicted_logS']].to_string(index=False))

if __name__=='__main__':main()
