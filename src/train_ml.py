"""Train-only fitting, validation-only selection, one locked test evaluation.
All features are generated from SMILES; external ESOL predicted column excluded.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.core import load_project, regression_metrics, dump_json, Y_COL


def make_model(kind, seed, threads, device='cpu'):
    imp=('impute',SimpleImputer(strategy='median', add_indicator=False,keep_empty_features=True))
    if kind=='dummy': return Pipeline([imp,('model',DummyRegressor(strategy='mean'))])
    if kind=='ridge': return Pipeline([imp,('scale',StandardScaler()),('model',Ridge(alpha=20))])
    if kind=='svr': return Pipeline([imp,('scale',StandardScaler()),('model',SVR(C=15,epsilon=0.12,gamma='scale'))])
    if kind=='rf': return Pipeline([imp,('model',RandomForestRegressor(n_estimators=350,min_samples_leaf=2,max_features=0.9,n_jobs=threads,random_state=seed))])
    if kind=='et': return Pipeline([imp,('model',ExtraTreesRegressor(n_estimators=350,min_samples_leaf=2,max_features=0.9,n_jobs=threads,random_state=seed))])
    xgbs={
       'xgb_smooth':dict(max_depth=3,n_estimators=600,learning_rate=0.025,subsample=0.85,colsample_bytree=0.85,min_child_weight=3,reg_lambda=4),
       'xgb_deep':dict(max_depth=5,n_estimators=450,learning_rate=0.035,subsample=0.8,colsample_bytree=0.85,min_child_weight=2,reg_lambda=8),
    }
    if kind in xgbs:
        return Pipeline([imp,('model',XGBRegressor(objective='reg:squarederror',tree_method='hist',device=device,
                 n_jobs=threads,random_state=seed,**xgbs[kind]))])
    raise ValueError(kind)


def candidates():
    # Prespecified, small candidate set; every test score withheld until selection.
    return [
        ('Dummy_train_mean','basic','dummy'),
        ('Ridge_basic','basic','ridge'),
        ('RF_basic','basic','rf'),
        ('RF_extended','extended','rf'),
        ('ExtraTrees_extended','extended','et'),
        ('SVR_extended','extended','svr'),
        ('XGB_basic_smooth','basic','xgb_smooth'),
        ('XGB_extended_smooth','extended','xgb_smooth'),
        ('XGB_extended_deep','extended','xgb_deep'),
        ('XGB_combined_smooth','combined','xgb_smooth'),
        ('XGB_combined_deep','combined','xgb_deep'),
    ]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True,help='Path containing prepared split/features')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--xgb-device',choices=['cpu','cuda'],default='cpu')
    args=ap.parse_args()
    p=Path(args.input)
    output=p/'ml';output.mkdir(exist_ok=True)
    df,split,features=load_project(p)
    y=df[Y_COL].to_numpy(float)
    tr=np.asarray(split['train'],dtype=int);va=np.asarray(split['val'],dtype=int);te=np.asarray(split['test'],dtype=int)
    records=[]; best=None; best_score=float('inf')
    for name,representation,kind in candidates():
        X=features[representation]
        model=make_model(kind,split['seed'],args.threads,args.xgb_device)
        model.fit(X[tr],y[tr])
        pv=np.asarray(model.predict(X[va]),dtype=float).reshape(-1)
        metrics=regression_metrics(y[va],pv)
        records.append({'model':name,'representation':representation,'algorithm':kind,**metrics})
        print(f'[ML VALIDATION] {name:25s} RMSE={metrics["RMSE"]:.4f} MAE={metrics["MAE"]:.4f} R2={metrics["R2"]:.3f}',flush=True)
        if kind!='dummy' and metrics['RMSE'] < best_score:
            best_score=metrics['RMSE'];best=(name,representation,kind,model)
    leaderboard=pd.DataFrame(records).sort_values('RMSE')
    leaderboard.to_csv(output/'validation_leaderboard.csv',index=False)
    best_name,repr_name,kind,model=best
    # Test was NOT used for model/feature/hyperparameter selection.
    test_pred=np.asarray(model.predict(features[repr_name][te])).reshape(-1)
    test_metrics=regression_metrics(y[te],test_pred)
    dummy=make_model('dummy',split['seed'],args.threads)
    dummy.fit(features['basic'][tr],y[tr])
    dummy_pred=dummy.predict(features['basic'][te]).reshape(-1)
    dummy_metrics=regression_metrics(y[te],dummy_pred)
    joblib.dump({'model':model,'representation':repr_name,'feature_set':repr_name,'name':best_name},output/'selected_model.joblib')
    preds=df.iloc[te][['Compound ID','smiles','canonical_smiles',Y_COL]].copy()
    preds['predicted_logS']=test_pred
    preds['residual_actual_minus_pred']=y[te]-test_pred
    preds.to_csv(output/'test_predictions.csv',index=False)
    info={'chosen_by':'minimum validation RMSE (test unseen)','winner':best_name,
          'representation':repr_name,'feature_dim':int(features[repr_name].shape[1]),
          'validation_RMSE':float(best_score),'test':test_metrics,'dummy_test':dummy_metrics,
          'train_samples':len(tr),'val_samples':len(va),'test_samples':len(te),
          'xgb_device':args.xgb_device,
          'warning':'TEST ONLY EVALUATED AFTER MODEL SELECTION. Do not alter model based on test results.'}
    dump_json(output/'test_metrics.json',info)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    axes[0].scatter(y[te],test_pred,s=19,alpha=0.65)
    a=min(y[te].min(),test_pred.min());b=max(y[te].max(),test_pred.max())
    axes[0].plot([a,b],[a,b],'k--',lw=1)
    axes[0].set(xlabel='Measured logS',ylabel='Predicted logS',title=f'{best_name}  RMSE={test_metrics["RMSE"]:.3f}')
    axes[1].scatter(test_pred,y[te]-test_pred,s=19,alpha=0.65)
    axes[1].axhline(0,color='black',ls='--',lw=1)
    axes[1].set(xlabel='Predicted logS',ylabel='Residual (actual - predicted)',title='Test residuals')
    fig.tight_layout();fig.savefig(output/'test_scatter_residual.png',dpi=180);plt.close(fig)
    top=leaderboard.iloc[:10].sort_values('RMSE')
    fig,ax=plt.subplots(figsize=(9,5.4));ax.barh(top['model'],top['RMSE']);ax.invert_yaxis()
    ax.set(xlabel='Validation RMSE (logS)',title='Validation-only model comparison')
    fig.tight_layout();fig.savefig(output/'validation_comparison.png',dpi=180);plt.close(fig)
    # Feature importance for the selected tree model (training model only).
    estimator=model.named_steps['model']
    if hasattr(estimator,'feature_importances_'):
        names=json.loads((p/'feature_names.json').read_text())[repr_name]
        importance=pd.DataFrame({'feature':names,'importance':estimator.feature_importances_}).sort_values('importance',ascending=False)
        importance.to_csv(output/'feature_importance.csv',index=False)
        top_imp=importance.head(15).iloc[::-1]
        fig,ax=plt.subplots(figsize=(8,5));ax.barh(top_imp.feature,top_imp.importance)
        ax.set(xlabel='Model feature importance (training-derived)',title=best_name)
        fig.tight_layout();fig.savefig(output/'feature_importance.png',dpi=180);plt.close(fig)
    print('[ML FINAL TEST]',json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
