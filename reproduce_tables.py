"""Recompute every PGS-PCQA number of the intra-dataset table (Table 1) and of the cross-dataset tables (Table 2, the
PMP-PCQA source-target table of the supplementary material) from the shipped per-sample
predictions, and compare them with the values printed in the paper.

usage: python reproduce_tables.py                     (reads ./predictions, prints all tables, exit code 1 on a mismatch)
       python reproduce_tables.py --intra_only --intra_dir <dir with preds_besttest_fold*.csv> --intra_db <DB>
                                                      (metrics of your own k-fold predictions)

Metrics (metrics.py): intra-dataset = per test fold SRCC on the raw predictions, PLCC / RMSE after the 5-parameter
logistic mapping, averaged over the folds. Cross-dataset = SRCC over all target samples of the single model per source.
Exact agreement in the 3rd-4th decimal of PLCC / RMSE needs scipy==1.15.3 (the logistic fit is version-sensitive).
"""
import argparse, glob, os, sys, warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import cv_metrics  # noqa: E402

warnings.filterwarnings("ignore")   # overflow / covariance warnings of the logistic fit are harmless

CSV_DIR = {'SJTU': 'sjtu_data_info', 'WPC': 'wpc_data_info', 'BASICS': 'basics_data_info', 'LS_PCQA_part': 'ls_pcqa_data_info'}
INTRA_DIR = {'SJTU': 'SJTU', 'WPC': 'WPC', 'BASICS': 'BASICS', 'LS_PCQA_part': 'LS'}
SHOW = {'SJTU': 'SJTU-PCQA', 'WPC': 'WPC', 'BASICS': 'BASICS', 'LS_PCQA_part': 'LS-PCQA'}
SHORT = {'SJTU': 'SJTU', 'WPC': 'WPC', 'BASICS': 'BASICS', 'LS_PCQA_part': 'LS'}

# values printed in the paper (SRCC / PLCC / RMSE; RMSE of WPC with 2 decimals)
PAPER_INTRA = {'SJTU': (0.955, 0.969, 0.570), 'WPC': (0.932, 0.933, 8.16),
               'BASICS': (0.888, 0.929, 0.384), 'LS_PCQA_part': (0.791, 0.802, 0.487)}
SOURCES = ('SJTU', 'WPC', 'LS_PCQA_part')   # one model per source, used for all targets
PAPER_TAB2 = {('SJTU', 'WPC'): 0.709, ('SJTU', 'BASICS'): 0.751, ('WPC', 'SJTU'): 0.799, ('WPC', 'BASICS'): 0.667}
PAPER_PMP = {('SJTU', 'LS_PCQA_part'): 0.438, ('SJTU', 'BASICS'): 0.751, ('LS_PCQA_part', 'SJTU'): 0.804,
             ('LS_PCQA_part', 'BASICS'): 0.623, ('LS_PCQA_part', 'WPC'): 0.795, ('WPC', 'LS_PCQA_part'): 0.590}
PAPER_CENTRE = {('SJTU', 'WPC'): 0.677}   # main text: SJTU->WPC with the centre crop only (reported model)

mismatches = []


def check(label, got, want, nd=3):
    ok = round(got, nd) == round(want, nd)
    if not ok:
        mismatches.append(f'{label}: recomputed {got:.{nd + 1}f}, paper {want:.{nd}f}')
    return 'ok' if ok else 'MISMATCH'


def load_intra(d):
    files = sorted(glob.glob(os.path.join(d, 'preds_besttest_fold*.csv')))
    assert files, f'no preds_besttest_fold*.csv in {d}'
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def check_folds(db, df):
    """every fold's predictions cover exactly the samples of the shipped split file test_<k>.csv"""
    for k, g in df.groupby('fold'):
        split = pd.read_csv(os.path.join(HERE, 'pgs_pcqa', 'csvfiles', CSV_DIR[db], f'test_{k}.csv'))
        assert sorted(g.name) == sorted(split.name), f'{db} fold {k}: predictions do not match test_{k}.csv'


def cross_pred(src, tgt, mode='fg4x4'):
    df = pd.read_csv(os.path.join(HERE, 'predictions', 'cross', f'{SHORT[src]}_source', f'{tgt}_{mode}.csv'))
    split = pd.read_csv(os.path.join(HERE, 'pgs_pcqa', 'csvfiles', CSV_DIR[tgt], 'test_1_xtgt.csv'))
    assert sorted(df.name) == sorted(split.name), f'{src}->{tgt}: predictions do not match the target split'
    return float(spearmanr(df.mos, df.pred)[0]), len(df)


def intra_table(dirs):
    print('Table 1 (intra-dataset, PGS-PCQA row): SRCC / PLCC / RMSE, mean over test folds, best test epoch per fold')
    print(f'  {"dataset":10s} {"folds":>5s} {"samples":>7s}   {"SRCC":>6s} {"PLCC":>6s} {"RMSE":>6s}   paper                check')
    for db, d in dirs.items():
        df = load_intra(d); check_folds(db, df); m = cv_metrics(df)
        rd = 2 if db == 'WPC' else 3
        got = (m['SRCC'], m['PLCC'], m['RMSE'])
        line = f'  {SHOW[db]:10s} {df.fold.nunique():5d} {len(df):7d}   {got[0]:.3f}  {got[1]:.3f}  {got[2]:.{rd}f}'
        if db in PAPER_INTRA and os.path.abspath(d).startswith(os.path.join(HERE, 'predictions')):
            p = PAPER_INTRA[db]
            st = [check(f'Table 1 {SHOW[db]} {n}', g, w, nd) for n, g, w, nd in zip(('SRCC', 'PLCC', 'RMSE'), got, p, (3, 3, rd))]
            line += f'   {p[0]:.3f} / {p[1]:.3f} / {p[2]:.{rd}f}   {"ok" if all(s == "ok" for s in st) else "MISMATCH"}'
        print(line + f'      (unrounded {got[0]:.4f} / {got[1]:.4f} / {got[2]:.4f})')


def cross_tables():
    print('\nTable 2 (cross-dataset SRCC, one model per source, foreground-weighted 4x4 testing)')
    for (s, t), w in PAPER_TAB2.items():
        r, n = cross_pred(s, t)
        print(f'  {SHORT[s]:>6s} -> {SHORT[t]:6s}  {r:.3f}  (paper {w:.3f}, {n} target samples)  {check(f"Table 2 {s}->{t}", r, w)}')
    print('\nSupplementary, source-target pairs of PMP-PCQA (same models)')
    for (s, t), w in PAPER_PMP.items():
        r, n = cross_pred(s, t)
        print(f'  {SHORT[s]:>6s} -> {SHORT[t]:6s}  {r:.3f}  (paper {w:.3f}, {n} target samples)  {check(f"PMP table {s}->{t}", r, w)}')
    print('\nCentre crop only (same weights)')
    for s in SOURCES:
        for t in ('SJTU', 'WPC', 'BASICS', 'LS_PCQA_part'):
            if s == t:
                continue
            r, _ = cross_pred(s, t, 'centre'); r4, _ = cross_pred(s, t)
            note = f'  (paper {PAPER_CENTRE[(s, t)]:.3f}) {check(f"centre {s}->{t}", r, PAPER_CENTRE[(s, t)])}' if (s, t) in PAPER_CENTRE else ''
            print(f'  {SHORT[s]:>6s} -> {SHORT[t]:6s}  centre {r:.3f} | fg 4x4 {r4:.3f}{note}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--intra_only', action='store_true')
    ap.add_argument('--intra_dir', default='')
    ap.add_argument('--intra_db', default='')
    a = ap.parse_args()
    print(f'scipy {scipy.__version__}' + ('' if scipy.__version__ == '1.15.3' else
          '  (WARNING: the paper numbers were computed with scipy 1.15.3; PLCC / RMSE may differ in the 3rd-4th decimal)'))
    if a.intra_dir:
        intra_table({a.intra_db: a.intra_dir})
    else:
        intra_table({db: os.path.join(HERE, 'predictions', 'intra', INTRA_DIR[db]) for db in PAPER_INTRA})
    if not a.intra_only:
        cross_tables()
    print('\nall recomputed values match the paper' if not mismatches else '\nMISMATCHES:\n  ' + '\n  '.join(mismatches))
    sys.exit(1 if mismatches else 0)
