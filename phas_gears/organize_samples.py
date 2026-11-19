"""Organize downloaded training samples into train/val/test folders for classifier training.

This is a re-implementation of ``organize_samples.py`` from the tanglethon-ipynb repository.
It reads the output of the ``clone_samples`` gear, maps PHAS labels to classes using a JSON
file, splits the samples into folds, and creates the folder structure expected by
``torchvision.datasets.ImageFolder`` (and ``tangle_cnn`` training)::

    <outdir>/patches/<fold>/<class>/<sample_id>.png   (symbolic links to the patches)

The split keeps all samples from a specimen (or slide) in the same fold, so that validation
and test performance are not inflated by near-duplicate samples from the same tissue, and it
keeps the class proportions in each fold close to those of the whole dataset.

The class JSON file is a list of classes, each with a list of regular expressions that are
matched against the full PHAS label name. A label is assigned to the first class that matches.
Classes with ``"ignore": 1`` are left out::

    [
      { "classname": "tangle", "labels": [ "tangle", "pretangle" ] },
      { "classname": "ignore", "labels": [ "ask_dave", "ambiguous" ], "ignore": 1 },
      { "classname": "nontangle", "labels": [ ".*" ] }
    ]
"""

import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

FOLDS = ('train', 'val', 'test')


class ClassMap:
    """Maps PHAS label names to class names using regular expressions from a JSON file."""

    def __init__(self, filename):
        with open(filename, 'rt') as f:
            self.spec = json.load(f)
        self.classes = []
        for c in self.spec:
            if 'classname' not in c or 'labels' not in c:
                raise ValueError(f'Each class in {filename} must have "classname" and "labels" entries: {c}')
            c['_re'] = [re.compile(p) for p in c['labels']]
            c['_ignore'] = bool(c.get('ignore', 0))

    def __call__(self, label):
        """Returns (classname, ignore) for a label, or (None, None) if no class matches."""
        for c in self.spec:
            if any(r.fullmatch(label) for r in c['_re']):
                return c['classname'], c['_ignore']
        return None, None


def assign_groups(counts: pd.DataFrame, fractions: dict, rng):
    """Assign groups (specimens, slides or samples) to folds so that every fold receives
    approximately the requested fraction of the samples of every class.

    Groups are assigned greedily, largest first (ties broken randomly), each to the fold
    where it most reduces the deviation from the per-class targets. Deviations are
    normalized by the class totals so that small classes count as much as large ones.

    Args:
        counts: DataFrame indexed by group, one column per class, with sample counts
        fractions: dict mapping fold name to the fraction of samples it should receive
        rng: numpy random generator

    Returns:
        dict mapping group to fold name
    """
    folds = [f for f, v in fractions.items() if v > 0]
    totals = counts.sum(axis=0).to_numpy(dtype=float)
    norm = np.maximum(totals, 1.0) ** 2
    target = np.array([fractions[f] * totals for f in folds])
    have = np.zeros_like(target)

    # Largest groups first, random order among equal sizes
    order = rng.permutation(len(counts))
    sizes = counts.sum(axis=1).to_numpy()[order]
    order = order[np.argsort(-sizes, kind='stable')]

    result = {}
    for i in order:
        g = counts.iloc[i].to_numpy(dtype=float)
        cost = (((have + g - target) ** 2 - (have - target) ** 2) / norm).sum(axis=1)
        k = int(np.argmin(cost))
        have[k] += g
        result[counts.index[i]] = folds[k]
    return result


def read_clone_manifest(indir):
    """Read the manifest written by clone_samples and locate the patch files."""
    df = pd.read_csv(os.path.join(indir, 'manifest.csv'),
                     dtype={'slide_name': str, 'specimen_private': str, 'label_name': str})
    if 'patch' not in df.columns:
        # Manifest from the older clone_samples.sh script
        df['patch'] = [os.path.join('all_patches', f'{i}.png') for i in df.id]
    df['patch_abs'] = [os.path.abspath(os.path.join(indir, p)) for p in df.patch]
    return df


def make_training_manifest(df):
    """Make w, h and mpp_x/mpp_y in the manifest describe the patch as saved, which is what
    tangle_cnn's ImageFolderWithBBox expects. When clone_samples resampled the patches (-r),
    the original values (in slide pixels / slide spacing) are kept in *_slide columns."""
    df = df.copy()
    if 'resolution' in df.columns and 'box_w' in df.columns:
        res = pd.to_numeric(df['resolution'], errors='coerce')
        rs = res.notna()
        if rs.any():
            for k in ('w', 'h', 'mpp_x', 'mpp_y'):
                if k in df.columns:
                    df[f'{k}_slide'] = df[k]
            df.loc[rs, 'w'] = df.loc[rs, 'box_w']
            df.loc[rs, 'h'] = df.loc[rs, 'box_h']
            # Slide spacing on PHAS (and in tangle_cnn) is in mm
            for k in ('mpp_x', 'mpp_y'):
                if k in df.columns:
                    df.loc[rs, k] = res[rs] / 1000.0
    return df


def clear_output(patch_dir):
    """Remove a previously generated patches folder. Only symbolic links and directories are
    removed; if a regular file is found, nothing is deleted."""
    if not os.path.exists(patch_dir):
        return
    for root, dirs, files in os.walk(patch_dir):
        for f in files:
            if not os.path.islink(os.path.join(root, f)):
                sys.exit(f'Error: {patch_dir} contains a regular file ({os.path.join(root, f)}); '
                         f'refusing to delete it. Is the output folder correct?')
    shutil.rmtree(patch_dir)


def organize_samples_main(args):
    rng = np.random.default_rng(args.seed)

    if os.path.realpath(args.outdir) == os.path.realpath(args.input):
        sys.exit('The output folder must be different from the input folder')

    # Read the samples and map labels to classes
    df = read_clone_manifest(args.input)
    print(f'Read {len(df)} samples from {args.input}')
    cmap = ClassMap(args.classes)
    mapped = {lab: cmap(lab) for lab in df.label_name.unique()}
    unmatched = sorted(lab for lab, (c, _) in mapped.items() if c is None)
    if unmatched:
        msg = f'Labels not matched by any class in {args.classes}: {unmatched}'
        if not args.drop_unmatched:
            sys.exit(f'Error: {msg}. Add them to a class (or an ignored class), or use --drop-unmatched')
        print(f'Warning: {msg}; dropping these samples')
    df['class'] = df.label_name.map(lambda lab: mapped[lab][0])
    df['ignore'] = df.label_name.map(lambda lab: mapped[lab][1])

    # Show how labels map to classes
    print('Label to class mapping:')
    lab_tab = df.groupby(['label_name'], dropna=False).agg(
        n=('id', 'size'), cls=('class', 'first'), ign=('ignore', 'first')).reset_index()
    for r in lab_tab.sort_values(['cls', 'label_name'], na_position='last').itertuples():
        cls = '(unmatched)' if r.cls is None or pd.isna(r.cls) else (f'{r.cls} (ignored)' if r.ign else r.cls)
        print(f'  {r.label_name:30s} {r.n:7d}  -> {cls}')

    df = df.loc[df['class'].notna() & ~df['ignore'].astype(bool)].copy()

    # Exclusions and missing patch files
    if args.exclude_specimens:
        df = df.loc[~df.specimen_private.isin(args.exclude_specimens)]
    exists = df.patch_abs.map(os.path.exists)
    if (~exists).any():
        print(f'Warning: {(~exists).sum()} samples have no patch file in {args.input} and are dropped')
        df = df.loc[exists]
    if len(df) == 0:
        sys.exit('Error: no samples left to organize')
    classes = sorted(df['class'].unique())

    # Assign samples to folds. Explicitly listed specimens go first.
    group_col = {'specimen': 'specimen_private', 'slide': 'slide_name', 'sample': 'id'}[args.split_by]
    df['fold'] = None
    for fold, spec in (('test', args.test_specimens), ('val', args.val_specimens)):
        if spec:
            unknown = sorted(set(spec) - set(df.specimen_private))
            if unknown:
                print(f'Warning: {fold} specimens not found among the samples: {unknown}')
            df.loc[df.fold.isna() & df.specimen_private.isin(spec), 'fold'] = fold

    # Remaining samples are split by group according to the fractions
    frac = {'test': 0.0 if args.test_specimens else args.test_fraction,
            'val': 0.0 if args.val_specimens else args.val_fraction}
    frac['train'] = 1.0 - frac['test'] - frac['val']
    if frac['train'] <= 0:
        sys.exit('Error: --val-fraction and --test-fraction leave no samples for training')
    pool = df.loc[df.fold.isna()]
    if frac['val'] > 0 or frac['test'] > 0:
        counts = pd.crosstab(pool[group_col], pool['class'])
        n_folds = sum(1 for v in frac.values() if v > 0)
        if len(counts) < 3 * n_folds:
            print(f'Warning: only {len(counts)} {args.split_by}s to split into {n_folds} folds; '
                  f'fold sizes and class proportions may be far from the requested fractions')
        assignment = assign_groups(counts, frac, rng)
        df.loc[pool.index, 'fold'] = pool[group_col].map(assignment)
    else:
        df.loc[pool.index, 'fold'] = 'train'

    n_before = pd.crosstab(df['class'], df['fold']).reindex(index=classes, columns=list(FOLDS), fill_value=0)

    # Per-class caps, applied by random subsampling
    def subsample(d, n):
        return d if len(d) <= n else d.sample(n=n, random_state=int(rng.integers(2**31)))

    keep = []
    for (cls, fold), d in df.groupby(['class', 'fold']):
        cap = {'train': args.max_train, 'val': args.max_val, 'test': args.max_test}[fold]
        keep.append(subsample(d, cap) if cap else d)
    df = pd.concat(keep)

    if args.balance_val:
        nv = df.loc[df.fold == 'val', 'class'].value_counts()
        if len(nv):
            n_min = int(nv.reindex(classes, fill_value=0).min())
            df = pd.concat([d if f != 'val' else subsample(d, n_min)
                            for (c, f), d in df.groupby(['class', 'fold'])])

    # Summary table
    n_after = pd.crosstab(df['class'], df['fold']).reindex(index=classes, columns=list(FOLDS), fill_value=0)
    groups = df.groupby('fold')[group_col].nunique().reindex(list(FOLDS), fill_value=0)
    print(f'\n{"class":24s}{"available":>10s}' + ''.join(f'{f:>9s}' for f in FOLDS))
    for c in classes:
        print(f'{c:24s}{n_before.loc[c].sum():10d}' + ''.join(f'{n_after.loc[c, f]:9d}' for f in FOLDS))
    print(f'{"total":24s}{n_before.values.sum():10d}' + ''.join(f'{n_after[f].sum():9d}' for f in FOLDS))
    print(f'{args.split_by + "s":24s}{"":10s}' + ''.join(f'{groups[f]:9d}' for f in FOLDS))

    # Every class must be present in train and val, otherwise ImageFolder assigns different
    # class indices to the two folds
    used_folds = [f for f in FOLDS if n_after[f].sum() > 0]
    for f in used_folds:
        missing = [c for c in classes if n_after.loc[c, f] == 0]
        if missing:
            msg = f'classes {missing} have no samples in the {f} fold'
            if f in ('train', 'val'):
                sys.exit(f'Error: {msg}. Class indices would not match between folds. '
                         f'Adjust the split (e.g., a different --seed or --split-by, or merge classes)')
            print(f'Warning: {msg}')

    # Build the output folder
    os.makedirs(args.outdir, exist_ok=True)
    patch_dir = os.path.join(args.outdir, 'patches')
    clear_output(patch_dir)
    for f in used_folds:
        for c in classes:
            os.makedirs(os.path.join(patch_dir, f, c), exist_ok=True)
    for sid, fold, cls, src in zip(df.id, df.fold, df['class'], df.patch_abs):
        link_dir = os.path.join(patch_dir, fold, cls)
        os.symlink(os.path.relpath(src, link_dir), os.path.join(link_dir, f'{sid}.png'))

    # Manifests and settings, so the split can be reproduced
    df_out = make_training_manifest(df.drop(columns=['patch_abs', 'ignore']).sort_values('id'))
    df_out.to_csv(os.path.join(args.outdir, 'split_manifest.csv'), index=False)
    n_after.to_csv(os.path.join(args.outdir, 'split_summary.csv'))
    shutil.copyfile(args.classes, os.path.join(args.outdir, 'classes.json'))
    with open(os.path.join(args.outdir, 'organize_args.json'), 'wt') as f:
        json.dump({k: v for k, v in vars(args).items() if k != 'action'}, f, indent=2)
    print(f'\nWrote {len(df)} links to {patch_dir}')
    print(f'Use {os.path.join(args.outdir, "split_manifest.csv")} as the manifest for training')
