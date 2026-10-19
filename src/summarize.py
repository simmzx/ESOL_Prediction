import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);args=p.parse_args()
    folder=Path(args.input)
    audit=json.loads((folder/'audit.json').read_text());sp=json.loads((folder/'split.json').read_text())
    ml_file=folder/'ml'/'test_metrics.json'
    ml=json.loads(ml_file.read_text()) if ml_file.exists() else None
    rnn_file=folder/'rnn'/'all_seed_metrics.csv'
    rnn=pd.read_csv(rnn_file) if rnn_file.exists() else None
    lines=['# ESOL Lab 2 — auto-generated experimental result summary','',
    '## 1. Data preparation and cleaning',
    f'- Original and retained labeled measurements: {audit["raw_rows"]} / {audit["valid_rows"]}.',
    f'- Canonical SMILES groups with duplicates: {audit["duplicate_canonical_groups"]}; conflicting labels: {audit["conflicting_label_groups"]}.',
    f'- Target label: `measured log solubility in mols per litre`; external predicted-label column excluded.',
    f'- SHA256: `{audit["source_sha256"]}`.', '',
    '## 2. Split and reproducibility',
    f'- Split: {sp["split"]}, seed {sp["seed"]}, train/val/test: {sp["counts"]}.',
    '- Same canonical structure cannot cross subsets; for scaffold split, Murcko scaffolds are disjoint.',
    '- All feature preprocessing and target normalization fit on training rows only.', '',
    '## 3. Model construction and validation',
    '- Baseline RNN: Embedding(64) → BiLSTM(64) → BiLSTM(32) → Dense(64) → Dropout(0.3) → linear regression.',
    '- Improved RNN: Embedding(64) → BiGRU(64) → masked attention → regression head.',
    '- ML candidates: Dummy, Ridge, RF, ExtraTrees, SVR, XGBoost; representations: 7/28 2D descriptors ± Morgan(1024).',
    '- ML winner selected by minimum validation RMSE; no test-driven tuning.', '']
    benchmark_rows=[]
    if ml:
        benchmark_rows.append({'model':ml['winner']+' (validation selected)','MAE':ml['test']['MAE'],'RMSE':ml['test']['RMSE'],'R2':ml['test']['R2'],'protocol':'single fit'})
        benchmark_rows.append({'model':'Dummy mean','MAE':ml['dummy_test']['MAE'],'RMSE':ml['dummy_test']['RMSE'],'R2':ml['dummy_test']['R2'],'protocol':'single fit'})
        lines+=['## 4. ML held-out test',
                f'- Validation winner: **{ml["winner"]}** using {ml["representation"]}.',
                f'- Winner test: MAE={ml["test"]["MAE"]:.4f}, RMSE={ml["test"]["RMSE"]:.4f}, R²={ml["test"]["R2"]:.4f}.',
                f'- Dummy test: MAE={ml["dummy_test"]["MAE"]:.4f}, RMSE={ml["dummy_test"]["RMSE"]:.4f}, R²={ml["dummy_test"]["R2"]:.4f}.',
                f'- RMSE reduction vs dummy: {(1-ml["test"]["RMSE"]/ml["dummy_test"]["RMSE"])*100:.1f}%.','']
    if rnn is not None:
        lines+=['## 5. RNN held-out test (all predeclared seeds)']
        for name,g in rnn.groupby('architecture'):
            benchmark_rows.append({'model':name,'MAE':g.MAE.mean(),'RMSE':g.RMSE.mean(),'R2':g.R2.mean(),'protocol':f'{len(g)} prespecified seed(s)'})
            lines.append(f'- {name}: RMSE={g.RMSE.mean():.4f} ± {g.RMSE.std(ddof=1):.4f} (n={len(g)}); MAE={g.MAE.mean():.4f}; R²={g.R2.mean():.4f}.') if len(g)>1 else lines.append(f'- {name}: RMSE={g.RMSE.mean():.4f} (one seed); MAE={g.MAE.mean():.4f}; R²={g.R2.mean():.4f}.')
        if ml is not None and 'teacher_bilstm' in rnn.architecture.values:
            base=rnn[rnn.architecture=='teacher_bilstm'].RMSE.mean()
            ratio=(1-ml['test']['RMSE']/base)*100
            lines+=['', f'- ML winner RMSE change vs teacher-style BiLSTM mean: **{ratio:+.2f}% reduction** (positive = improvement).',
                    '- Note: ML and RNN seed aggregation differ; report protocol clearly.']
    if benchmark_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        table=pd.DataFrame(benchmark_rows)
        table.to_csv(folder/'heldout_benchmark.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,4.4))
        plot=table.sort_values('RMSE')
        ax.barh(plot['model'],plot['RMSE']);ax.invert_yaxis()
        ax.set(xlabel='Test RMSE (logS)',title=f'Held-out comparison, {sp["split"]} split (RNN means if repeated)')
        fig.tight_layout();fig.savefig(folder/'heldout_benchmark.png',dpi=180);plt.close(fig)
    lines+=['','## 6. Plots, review, and limitations',
            '- Plots under `distribution.png`, `ml/*.png`, `rnn/*/seed_*/*.png`.',
            '- Validation leaderboard and feature importance under `ml/`.',
            '- Strict scaffold performance is a separate generalization question; random-split results are not scaffold scores.',
            '- This summary is a drafting aid, NOT a finished report. Add code excerpts, method rationale, experiment logs and references.',
            '- Teacher baseline is a PyTorch reproduction of PPT architecture, not identical TensorFlow/Keras execution.']
    (folder/'REPORT_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines))

if __name__=='__main__':main()
