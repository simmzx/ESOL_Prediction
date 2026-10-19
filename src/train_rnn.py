"""PPT-inspired BiLSTM and improved BiGRU + masked attention (PyTorch).
Source of truth for labels/splits: output of src.prepare. GPU selected by
CUDA_VISIBLE_DEVICES=1 in shell; visible cuda:0 maps to physical GPU 1.
"""
import argparse
import copy
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.core import load_project,regression_metrics,dump_json,Y_COL

PAD=0
UNK=1

class TeacherBiLSTM(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,64,padding_idx=PAD)
        self.lstm1=nn.LSTM(64,64,batch_first=True,bidirectional=True)
        self.lstm2=nn.LSTM(128,32,batch_first=True,bidirectional=True)
        self.linear=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.30),nn.Linear(64,1))
    def forward(self,tokens,lengths):
        x=self.embedding(tokens)
        p=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        p,_=self.lstm1(p)
        _,(h,_)=self.lstm2(p)
        features=torch.cat((h[-2],h[-1]),dim=1)
        return self.linear(features).squeeze(-1)

class BigruAttention(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,64,padding_idx=PAD)
        self.gru=nn.GRU(64,64,bidirectional=True,batch_first=True)
        self.attn=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        self.head=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Dropout(.35),nn.Linear(64,1))
    def forward(self,tokens,lengths):
        x=self.embedding(tokens)
        p=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        p,h=self.gru(p)
        seq,_=nn.utils.rnn.pad_packed_sequence(p,batch_first=True,total_length=tokens.size(1))
        score=self.attn(seq).squeeze(-1)
        valid=torch.arange(tokens.size(1),device=tokens.device)[None,:] < lengths[:,None]
        weights=score.masked_fill(~valid,-1e9).softmax(dim=1)
        context=(seq*weights.unsqueeze(-1)).sum(dim=1)
        end=torch.cat([h[-2],h[-1]],dim=1)
        return self.head(torch.cat([context,end],dim=1)).squeeze(-1)


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def tokenize(df,split):
    # Train-only vocabulary; UNK handles tokens unseen in val/test. Right-padding.
    tr=np.array(split['train']);train_s=df.iloc[tr].smiles.astype(str).tolist()
    vocab={c:i+2 for i,c in enumerate(sorted(set(''.join(train_s))))}
    mx=max(len(s) for s in train_s)
    def enc(strings):
        arr=np.zeros((len(strings),mx),dtype=np.int64)
        lens=np.empty(len(strings),dtype=np.int64)
        clipped=0
        unknown=0
        for i,s in enumerate(strings):
            if len(s)>mx:clipped+=1
            seq=[vocab.get(c,UNK) for c in s[:mx]]
            unknown+=sum(q==UNK for q in seq)
            arr[i,:len(seq)]=seq;lens[i]=max(1,len(seq))
        return arr,lens,clipped,unknown
    tokens,lengths,clipped,unknown=enc(df.smiles.astype(str).tolist())
    return vocab,mx,tokens,lengths,clipped,unknown


def evaluate(model,idx,tokens,lengths,y,mean,std,device,batch_size):
    model.eval()
    ps=[]
    with torch.no_grad():
        for start in range(0,len(idx),batch_size):
            ix=idx[start:start+batch_size]
            tx=torch.from_numpy(tokens[ix]).to(device)
            ll=torch.from_numpy(lengths[ix]).to(device)
            ps.extend(model(tx,ll).cpu().numpy().tolist())
    return np.asarray(ps)*std+mean


def fit_one(architecture,seed,df,split,arrs,device,output,teacher_epochs,attention_epochs,patience,batch_size):
    set_seed(seed)
    tr=np.asarray(split['train']);va=np.asarray(split['val']);te=np.asarray(split['test'])
    y=df[Y_COL].to_numpy(float)
    vocab,mx,tokens,lengths,clipped,unknown=arrs
    mu=float(y[tr].mean()); sd=float(y[tr].std())
    if sd<=0:raise ValueError('Zero target variance')
    t=torch.from_numpy(tokens[tr])
    ll=torch.from_numpy(lengths[tr])
    yy=torch.tensor((y[tr]-mu)/sd,dtype=torch.float32)
    g=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(t,ll,yy),batch_size=batch_size,shuffle=True,generator=g,num_workers=0,pin_memory=(device.type=='cuda'))
    if architecture=='teacher_bilstm':
        model=TeacherBiLSTM(len(vocab)+2).to(device)
        max_epoch=teacher_epochs
        lr=1e-4
    else:
        model=BigruAttention(len(vocab)+2).to(device)
        max_epoch=attention_epochs
        lr=7e-4
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_func=nn.MSELoss()
    best=float('inf');best_state=None;best_epoch=0;hist=[];wait=0
    for epoch in range(1,max_epoch+1):
        model.train();running=0.;n=0
        for xt,lt,yt in loader:
            xt=xt.to(device,non_blocking=True);lt=lt.to(device,non_blocking=True);yt=yt.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred=model(xt,lt)
            loss=loss_func(pred,yt)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            running+=loss.item()*len(yt);n+=len(yt)
        pv=evaluate(model,va,tokens,lengths,y,mu,sd,device,batch_size)
        vm=regression_metrics(y[va],pv)
        log={'epoch':epoch,'train_MSE_scaled':running/n,'val_RMSE':vm['RMSE'],'val_MAE':vm['MAE']}
        hist.append(log)
        # Validation loss is used to select checkpoint. Test untouched.
        if vm['RMSE'] < best-1e-5:
            best=vm['RMSE'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());wait=0
        else:
            wait+=1
        if epoch==1 or epoch%5==0 or epoch==max_epoch:
            print(f'[RNN {architecture} seed={seed}] epoch={epoch}/{max_epoch} train_scaled_MSE={running/n:.4f} val_RMSE={vm["RMSE"]:.4f}',flush=True)
        if architecture!='teacher_bilstm' and wait>=patience:
            break
    model.load_state_dict(best_state)
    # TEST EVALUATED EXACTLY ONCE per preregistered seed after val checkpoint selection.
    pp=evaluate(model,te,tokens,lengths,y,mu,sd,device,batch_size)
    test_metrics=regression_metrics(y[te],pp)
    run_dir=output/architecture/f'seed_{seed}';run_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(hist).to_csv(run_dir/'learning_curve.csv',index=False)
    outpred=df.iloc[te][['Compound ID','smiles','canonical_smiles',Y_COL]].copy()
    outpred['predicted_logS']=pp
    outpred['residual_actual_minus_pred']=y[te]-pp
    outpred.to_csv(run_dir/'test_predictions.csv',index=False)
    torch.save({'architecture':architecture,'state_dict':model.cpu().state_dict(),
         'vocab':vocab,'max_length':mx,'y_mean':mu,'y_std':sd,'seed':seed},run_dir/'checkpoint.pt')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    axes[0].plot([r['epoch'] for r in hist],[r['train_MSE_scaled'] for r in hist],label='Train scaled MSE')
    axes[0].set(xlabel='Epoch',ylabel='Scaled MSE',title='Training loss')
    axes[1].plot([r['epoch'] for r in hist],[r['val_RMSE'] for r in hist],label='Validation RMSE')
    axes[1].axvline(best_epoch,ls='--',color='k',label='Best checkpoint')
    axes[1].set(xlabel='Epoch',ylabel='logS RMSE',title='Validation performance')
    axes[1].legend();fig.tight_layout();fig.savefig(run_dir/'learning_curve.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,5.2))
    ax.scatter(y[te],pp,alpha=.7,s=18);a=min(y[te].min(),pp.min());b=max(y[te].max(),pp.max())
    ax.plot([a,b],[a,b],'k--');ax.set(xlabel='Measured logS',ylabel='Predicted logS',title=f'{architecture} seed={seed} RMSE={test_metrics["RMSE"]:.3f}')
    fig.tight_layout();fig.savefig(run_dir/'test_scatter.png',dpi=160);plt.close(fig)
    record={'architecture':architecture,'seed':seed,'best_epoch':best_epoch,
        'train_epochs':len(hist),'val_best_RMSE':best,'test':test_metrics,
        'architecture_details':'teacher: Emb64-BiLSTM64-BiLSTM32-Dense64-Dropout0.3 or Emb64-BiGRU64-Attention-Dense64',
        'target_scaler':'train-only mean/std','token_vocab':'train-only character vocabulary',
        'device':str(device),'train_size':len(tr),'val_size':len(va),'test_size':len(te)}
    dump_json(run_dir/'metrics.json',record)
    print(f'[RNN TEST] {architecture} seed={seed}: {test_metrics}',flush=True)
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42])
    ap.add_argument('--teacher-epochs',type=int,default=30)
    ap.add_argument('--attention-epochs',type=int,default=80)
    ap.add_argument('--patience',type=int,default=12)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--require-gpu',action='store_true')
    ap.add_argument('--architecture',choices=['both','teacher_bilstm','bigru_attention'],default='both')
    args=ap.parse_args()
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU but no CUDA available. Check nvidia-smi and GPU-specific torch install.')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[DEVICE] CUDA_VISIBLE_DEVICES={os.getenv("CUDA_VISIBLE_DEVICES","NOT SET")}, torch={torch.__version__}, using={device}, available={torch.cuda.is_available()}',flush=True)
    p=Path(args.input);df,split,_=load_project(p)
    tokens=tokenize(df,split)
    vocab,mx,_,_,clip,unknown=tokens
    print(f'[TOKEN] train vocabulary {len(vocab)}, max len {mx}, clipped molecules {clip}, unknown characters {unknown}',flush=True)
    out=p/'rnn';out.mkdir(exist_ok=True)
    dump_json(out/'token_audit.json',{'vocab_size':len(vocab),'max_length':mx,'num_truncated':clip,'num_unknown_characters':unknown,'vocab':vocab})
    rows=[]
    archs=['teacher_bilstm','bigru_attention'] if args.architecture=='both' else [args.architecture]
    for arch in archs:
        for seed in args.seeds:
            rows.append(fit_one(arch,seed,df,split,tokens,device,out,args.teacher_epochs,args.attention_epochs,args.patience,args.batch_size))
    flat=[]
    for r in rows:
        flat.append({'architecture':r['architecture'],'seed':r['seed'],'best_epoch':r['best_epoch'],
              'val_RMSE':r['val_best_RMSE'],**r['test']})
    pd.DataFrame(flat).to_csv(out/'all_seed_metrics.csv',index=False)
    stats=pd.DataFrame(flat).groupby('architecture')[['MAE','RMSE','R2']].agg(['mean','std','count'])
    stats.to_csv(out/'seed_summary.csv')
    print('[RNN SUMMARY]',stats.to_string(),flush=True)

if __name__=='__main__':main()
