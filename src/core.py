"""Strict ESOL loading, leakage-resistant grouped split, RDKit descriptors.
No label-derived/external-prediction columns are ever used as features.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

RDLogger.DisableLog('rdApp.warning')
SMILES_COL = 'smiles'
Y_COL = 'measured log solubility in mols per litre'
LEAK_COL = 'ESOL predicted log solubility in mols per litre'

# Names and RDKit functions (physicochemically motivated, all generated from SMILES).
BASIC = {
    'MolWt': Descriptors.MolWt,
    'MolLogP': Descriptors.MolLogP,
    'TPSA': rdMolDescriptors.CalcTPSA,
    'HBD': Lipinski.NumHDonors,
    'HBA': Lipinski.NumHAcceptors,
    'RotBonds': Lipinski.NumRotatableBonds,
    'AromaticRings': Lipinski.NumAromaticRings,
}
EXTRA = {
    'RingCount': Lipinski.RingCount,
    'HeavyAtomCount': Lipinski.HeavyAtomCount,
    'FractionCSP3': Lipinski.FractionCSP3,
    'NumHeteroatoms': Lipinski.NumHeteroatoms,
    'NumAliphaticRings': Lipinski.NumAliphaticRings,
    'NumSaturatedRings': Lipinski.NumSaturatedRings,
    'NHOHCount': Lipinski.NHOHCount,
    'NOCount': Lipinski.NOCount,
    'MolMR': Descriptors.MolMR,
    'LabuteASA': rdMolDescriptors.CalcLabuteASA,
    'BertzCT': Descriptors.BertzCT,
    'BalabanJ': Descriptors.BalabanJ,
    'HallKierAlpha': Descriptors.HallKierAlpha,
    'Kappa1': Descriptors.Kappa1,
    'Kappa2': Descriptors.Kappa2,
    'Chi0v': Descriptors.Chi0v,
    'Chi1v': Descriptors.Chi1v,
    'NumValenceElectrons': Descriptors.NumValenceElectrons,
    'NumRadicalElectrons': Descriptors.NumRadicalElectrons,
    'MaxPartialCharge': Descriptors.MaxPartialCharge,
    'MinPartialCharge': Descriptors.MinPartialCharge,
}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if SMILES_COL not in df or Y_COL not in df:
        raise ValueError(f'Required columns: {SMILES_COL!r}, {Y_COL!r}; got {list(df.columns)}')
    if LEAK_COL in df:
        print('[GUARD] Excluding external ESOL predicted logS column from ALL features.')
    df = df.copy()
    raw_n = len(df)
    df[SMILES_COL] = df[SMILES_COL].astype('string').str.strip()
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors='coerce')
    valid = []
    canonical = []
    scaffold = []
    for s in df[SMILES_COL]:
        mol = Chem.MolFromSmiles(str(s)) if pd.notna(s) and str(s) else None
        ok = mol is not None
        valid.append(ok)
        canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True) if ok else None)
        scaffold.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True) if ok else None)
    df['canonical_smiles'] = canonical
    df['scaffold'] = scaffold
    good = pd.Series(valid, index=df.index) & np.isfinite(df[Y_COL].to_numpy(dtype=float))
    dropped = df.loc[~good].copy()
    df = df.loc[good].reset_index(drop=True)
    if len(df) < 100:
        raise ValueError('Fewer than 100 valid ESOL records.')
    labels = df.groupby('canonical_smiles')[Y_COL].agg(['size','min','max']).reset_index()
    duplicates = labels.loc[labels['size'] > 1].copy()
    duplicates['label_range'] = duplicates['max'] - duplicates['min']
    audit = {
        'raw_rows': int(raw_n), 'valid_rows': int(len(df)),
        'invalid_or_missing_rows': int(len(dropped)),
        'unique_canonical_structures': int(df.canonical_smiles.nunique()),
        'duplicate_canonical_groups': int(len(duplicates)),
        'extra_canonical_records': int(sum(duplicates['size'] - 1)),
        'conflicting_label_groups': int(sum(duplicates.label_range > 1e-10)),
        'max_duplicate_label_range': float(duplicates.label_range.max()) if len(duplicates) else 0.,
        'unique_scaffolds': int(df.scaffold.nunique()),
        'acyclic_empty_scaffold_records': int((df.scaffold == '').sum()),
        'target_min': float(df[Y_COL].min()), 'target_max': float(df[Y_COL].max()),
        'target_mean': float(df[Y_COL].mean()),
        'target_std': float(df[Y_COL].std()),
        'excluded_external_prediction_column': LEAK_COL,
        'rdkit_version': rdBase.rdkitVersion,
        'duplicate_policy': 'Retain original observed labels; canonical SMILES never cross split boundaries.',
    }
    return df, dropped, duplicates, audit


def _scaffold_split(df, seed):
    """Deterministic strict scaffold disjointness, largest groups train first.
    Acyclic molecules have empty scaffold, so all acyclic go to one partition.
    This dataset has huge scaffold groups; proportions only approximate.
    """
    groups = {}
    for i, g in enumerate(df.scaffold):
        groups.setdefault(g, []).append(i)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))  # shuffle stabilizes tie order
    N = len(df)
    train, test, val = [], [], []
    for k in keys:
        rows = groups[k]
        if len(train) < round(0.72*N):
            train.extend(rows)
        elif len(test) < round(0.20*N):
            test.extend(rows)
        else:
            val.extend(rows)
    return sorted(train), sorted(val), sorted(test)


def build_split(df, split='random', seed=42):
    indices = np.arange(len(df))
    if split == 'random':
        # GroupShuffleSplit operates on canonical structures; no molecular duplicate leakage.
        g = df.canonical_smiles.to_numpy()
        a, b = next(GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=seed).split(indices, groups=g))
        aa, vv = next(GroupShuffleSplit(test_size=0.10, n_splits=1, random_state=seed+1).split(a, groups=g[a]))
        tr, va, te = sorted(a[aa].tolist()), sorted(a[vv].tolist()), sorted(b.tolist())
        group_col = 'canonical_smiles'
    elif split == 'scaffold':
        tr, va, te = _scaffold_split(df, seed)
        group_col = 'scaffold'
    else:
        raise ValueError('split must be random or scaffold')
    if not tr or not va or not te:
        raise ValueError('One split is empty; cannot proceed.')
    sets = [set(df.iloc[s][group_col]) for s in (tr, va, te)]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]), 'Group leakage'
    # random splits also group canonical; scaffold includes canonical by definition
    canonical = [set(df.iloc[s].canonical_smiles) for s in (tr, va, te)]
    assert all(not canonical[i].intersection(canonical[j]) for i,j in [(0,1),(0,2),(1,2)])
    assert set(tr+va+te) == set(indices.tolist())
    return {'split':split, 'seed':seed, 'train':tr, 'val':va, 'test':te,
            'counts':{'train':len(tr),'val':len(va),'test':len(te)},
            'group_col':group_col}


def feature_matrix(df, fp_bits=1024):
    names = list(BASIC) + list(EXTRA)
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_bits)
    Xdesc = np.empty((len(df),len(names)), dtype=np.float32)
    Xfp = np.empty((len(df),fp_bits), dtype=np.float32)
    for i, s in enumerate(df.canonical_smiles):
        mol = Chem.MolFromSmiles(s)
        vals = []
        for key, fun in [*BASIC.items(),*EXTRA.items()]:
            try:
                f = float(fun(mol))
                vals.append(f if np.isfinite(f) else np.nan)
            except Exception:
                vals.append(np.nan)
        Xdesc[i] = vals
        Xfp[i] = morgan.GetFingerprintAsNumPy(mol).astype(np.float32)
    result = {'basic': Xdesc[:,:len(BASIC)], 'extended':Xdesc,
              'combined':np.concatenate((Xdesc,Xfp),axis=1), 'fingerprint':Xfp}
    fnames = {'basic':list(BASIC),'extended':names,
              'combined':names+[f'Morgan_{i}' for i in range(fp_bits)],
              'fingerprint':[f'Morgan_{i}' for i in range(fp_bits)]}
    return result,fnames


def regression_metrics(y, pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {'MAE':float(mean_absolute_error(y,pred)),
            'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
            'R2':float(r2_score(y,pred))}


def dump_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')


def load_project(project_dir):
    p=Path(project_dir)
    df=pd.read_csv(p/'clean_data.csv')
    split=json.loads((p/'split.json').read_text(encoding='utf8'))
    a=np.load(p/'features.npz')
    return df,split,{k:a[k] for k in a.files}
