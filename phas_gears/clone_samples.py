"""Download training sample patches (user-placed boxes) from a PHAS classifier training task.

This is a Python re-implementation and extension of ``clone_samples.sh`` from the
tanglethon-ipynb repository. In addition to the original functionality it supports

* resampling patches to a target resolution (microns per pixel),
* filtering samples by specimen, stain, label and user (creator/editor),
* recording the location of each sample's box within its patch in the manifest.

Coordinates on the server are in full-resolution (level 0) slide pixels. All patch
sizes and paddings given on the command line are in *output* pixels, i.e., pixels at
the target resolution.
"""

from phas.client.api import Client, DLTrainingTask, Slide
from phas.dltrain import get_sample_manifest_for_task
from phas.dzi import get_patch_endpoint
from PIL import Image
from io import BytesIO, StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import os
import sys
import threading


@dataclass
class PatchSpec:
    """A rectangular region around a sample to download from a slide, in level-0 pixel units."""
    sample_id: int
    slide_id: int
    x0: float
    y0: float
    width: int                  # Output width, pixels at target resolution
    height: int                 # Output height, pixels at target resolution
    scale: float                # Size of an output pixel in level-0 pixels

    @property
    def filename(self):
        return os.path.join('patches', f'{self.sample_id}.png')


def _parse_user_list(users):
    """Split a list of user identifiers into numeric ids and usernames."""
    if not users:
        return set(), set()
    ids = set(int(u) for u in users if str(u).isdigit())
    names = set(str(u) for u in users if not str(u).isdigit())
    return ids, names


class SampleCloner:
    """Downloads training sample patches from a PHAS DL training task."""

    def __init__(self, task: DLTrainingTask, resolution: Optional[float] = None,
                 patch_size: Optional[int] = None, padding: int = 0,
                 tile_size: int = 1024):
        self.task = task
        self.resolution = resolution
        self.patch_size = patch_size
        self.padding = padding
        self.tile_size = tile_size
        self._slides = {}
        self._slide_lock = threading.Lock()
        self._warned_upsample = False

    # ------------------------------------------------------------------
    # Sample listing and filtering
    # ------------------------------------------------------------------
    def load_samples(self, specimens=None, exclude_specimens=None, stains=None,
                     labels=None, exclude_labels=None, users=None, exclude_users=None,
                     max_per_label=None, seed=0, raw_manifest_file=None):
        """Read and filter the list of samples in the task.

        Returns:
            DataFrame with one row per sample
        """
        # Task-wide manifest from the server (slide names, labels, user names, mpp)
        r = self.task.client._get('dltrain', get_sample_manifest_for_task, task_id=self.task.task_id)
        if raw_manifest_file:
            with open(raw_manifest_file, 'wt') as f:
                f.write(r.text)
        df = pd.read_csv(StringIO(r.text), dtype={'slide_name': str, 'specimen_private': str,
                                                  'block_name': str, 'stain': str,
                                                  'creator': str, 'editor': str},
                         na_values=['None'])
        print(f'Server manifest lists {len(df)} samples')

        # Slides without metadata (pixel spacing) cannot be read from the server
        no_meta = df.mpp_x.isna()
        if no_meta.any():
            print(f'Warning: skipping {no_meta.sum()} samples on {df.loc[no_meta].slide_name.nunique()} '
                  f'slides for which the server has no slide metadata (slide file unavailable?)')
            df = df.loc[~no_meta]

        # Slide-level filters
        if specimens:
            df = df.loc[df.specimen_private.isin(specimens)]
        if exclude_specimens:
            df = df.loc[~df.specimen_private.isin(exclude_specimens)]
        if stains:
            df = df.loc[df.stain.str.lower().isin([s.lower() for s in stains])]

        # Label filters
        if labels:
            df = df.loc[df.label_name.isin(labels)]
        if exclude_labels:
            df = df.loc[~df.label_name.isin(exclude_labels)]

        # Map slide names to slide ids. The manifest only has slide names.
        sm = pd.DataFrame(self.task.slide_manifest())
        sm['slide_name'] = sm['slide_name'].astype(str)
        sm = sm.loc[sm.slide_name.isin(df.slide_name.unique())]
        name_to_id = dict(zip(sm.slide_name, sm.id))
        missing = set(df.slide_name.unique()) - set(name_to_id.keys())
        if missing:
            print(f'Warning: could not find slide ids for slides {sorted(missing)}, skipping their samples')
            df = df.loc[df.slide_name.isin(name_to_id.keys())]
        df = df.assign(slide_id=df.slide_name.map(name_to_id).astype(int))

        # Get numeric creator/editor ids from per-slide sample listings
        uid = []
        for slide_id in df.slide_id.unique():
            for s in self.task.slide_training_samples(int(slide_id)):
                uid.append((s['id'], s['creator'], s['editor']))
        df_uid = pd.DataFrame(uid, columns=['id', 'creator_id', 'editor_id'])
        df = df.merge(df_uid, on='id', how='left')

        # User filters. Users can be specified by numeric id or by username. To be included,
        # both creator and editor must be in the list. To be excluded, either one suffices.
        def user_match(ids, names, which):
            return df[f'{which}_id'].isin(ids) | df[which].isin(names)

        if users:
            ids, names = _parse_user_list(users)
            df = df.loc[user_match(ids, names, 'creator') & user_match(ids, names, 'editor')]
        if exclude_users:
            ids, names = _parse_user_list(exclude_users)
            df = df.loc[~(user_match(ids, names, 'creator') | user_match(ids, names, 'editor'))]

        # Random subset per label
        if max_per_label:
            df = df.groupby('label_name', group_keys=False).apply(
                lambda g: g.sample(n=min(len(g), max_per_label), random_state=seed))

        df = df.sort_values('id').reset_index(drop=True)
        print(f'{len(df)} samples remaining after filtering, '
              f'from {df.slide_id.nunique()} slides and {df.specimen_private.nunique()} specimens')
        if len(df):
            print('Samples per label:')
            print(df.label_name.value_counts().to_string())
        return df

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def get_slide(self, slide_id):
        """Get slide and its pyramid level downsample factors (cached, thread-safe)"""
        with self._slide_lock:
            if slide_id not in self._slides:
                slide = Slide(self.task, int(slide_id))
                try:
                    lds = list(slide.level_downsamples)
                except Exception as e:
                    print(f'Warning: could not read pyramid levels for slide {slide_id} ({e}), '
                          f'reading from full resolution level')
                    lds = [1.0]
                self._slides[slide_id] = (slide, lds)
            return self._slides[slide_id]

    def downsample_factor(self, mpp_mm):
        """Size of an output pixel in level-0 pixels. Slide spacing on PHAS is in mm."""
        if self.resolution is None:
            return 1.0
        return self.resolution / (1000.0 * mpp_mm)

    def make_patch_spec(self, row, d):
        """Patch around one sample, snapped to whole output pixels and centered on the box.

        Args:
            row: Sample (row of the samples data frame)
            d: Size of output pixel in level-0 pixels
        """
        cx, cy = row.x + 0.5 * row.w, row.y + 0.5 * row.h
        if self.patch_size:
            W = H = self.patch_size
        else:
            W = max(1, int(np.ceil(row.w / d + 2 * self.padding - 1e-6)))
            H = max(1, int(np.ceil(row.h / d + 2 * self.padding - 1e-6)))
        return PatchSpec(sample_id=int(row.id), slide_id=int(row.slide_id),
                         x0=float(cx - 0.5 * W * d), y0=float(cy - 0.5 * H * d),
                         width=W, height=H, scale=float(d))

    def plan_patches(self, df):
        """Compute the patch to download for each sample.

        Returns:
            (list of PatchSpec, DataFrame df with added patch/box columns)
        """
        specs = []
        for slide_id, df_slide in df.groupby('slide_id'):
            mpp_mm = float(df_slide.mpp_x.iloc[0])
            d = self.downsample_factor(mpp_mm)
            if d < 1.0 and not self._warned_upsample:
                print(f'Warning: target resolution {self.resolution}um is finer than slide '
                      f'resolution {1000 * mpp_mm:.4f}um (slide {slide_id}); patches will be upsampled')
                self._warned_upsample = True
            specs.extend(self.make_patch_spec(row, d) for row in df_slide.itertuples())

        df_patch = pd.DataFrame([{
            'id': s.sample_id, 'patch': s.filename,
            'patch_x0': s.x0, 'patch_y0': s.y0, 'patch_scale': s.scale,
            'patch_width': s.width, 'patch_height': s.height,
            'resolution': self.resolution if self.resolution else '',
        } for s in specs])
        df_out = df.merge(df_patch, on='id', how='left')

        # Location of the sample's box in the patch, in output pixel units
        df_out['box_x'] = ((df_out.x - df_out.patch_x0) / df_out.patch_scale).round(2)
        df_out['box_y'] = ((df_out.y - df_out.patch_y0) / df_out.patch_scale).round(2)
        df_out['box_w'] = (df_out.w / df_out.patch_scale).round(2)
        df_out['box_h'] = (df_out.h / df_out.patch_scale).round(2)
        df_out = df_out.round({'patch_x0': 2, 'patch_y0': 2, 'patch_scale': 6})
        return specs, df_out

    # ------------------------------------------------------------------
    # Downloading
    # ------------------------------------------------------------------
    def read_region(self, slide, ds, level, x, y, w, h):
        """Read a region at a pyramid level with downsample ds, (x,y) the top left corner in level-0 pixels."""
        out = Image.new('RGBA', (w, h))
        ts = self.tile_size
        for ty in range(0, h, ts):
            for tx in range(0, w, ts):
                tw, th = min(ts, w - tx), min(ts, h - ty)
                # The server reads the region starting at ctr - int(0.5 * size * ds)
                X = int(x + round(tx * ds))
                Y = int(y + round(ty * ds))
                r = self.task.client._get('dzi', get_patch_endpoint,
                                          project=slide.project, slide_id=slide.slide_id,
                                          resource='raw', level=level,
                                          ctrx=X + int(tw * 0.5 * ds), ctry=Y + int(th * 0.5 * ds),
                                          w=tw, h=th, format='png')
                tile = Image.open(BytesIO(r.content))
                tile.load()
                out.paste(tile.convert('RGBA'), (tx, ty))
        return out

    def download_patch(self, spec: PatchSpec, filename):
        slide, lds = self.get_slide(spec.slide_id)
        d = spec.scale

        # Pick the coarsest pyramid level that is at least as fine as the target resolution
        level = max([i for i, ds in enumerate(lds) if ds <= d * (1 + 1e-3)] or [0])
        ds = lds[level]

        # Region to read, padded by a pixel at the read level to support interpolation
        x0i, y0i = int(np.floor(spec.x0 - ds)), int(np.floor(spec.y0 - ds))
        x1, y1 = spec.x0 + spec.width * d, spec.y0 + spec.height * d
        wl = int(np.ceil((x1 + ds - x0i) / ds))
        hl = int(np.ceil((y1 + ds - y0i) / ds))
        img = self.read_region(slide, ds, level, x0i, y0i, wl, hl)

        # Resample the exact region onto the output grid
        box = ((spec.x0 - x0i) / ds, (spec.y0 - y0i) / ds, (x1 - x0i) / ds, (y1 - y0i) / ds)
        if (spec.width, spec.height) != (wl, hl) or any(abs(b - round(b)) > 1e-6 for b in box):
            img = img.resize((spec.width, spec.height), resample=Image.LANCZOS, box=box)

        # Areas outside of the slide come back transparent; composite onto white
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(bg, img).convert('RGB')

        tmp = filename + '.tmp.png'
        rgb.save(tmp)
        os.replace(tmp, filename)

    def download_all(self, specs, outdir, jobs=4, skip_existing=False):
        os.makedirs(os.path.join(outdir, 'patches'), exist_ok=True)

        todo = []
        for spec in specs:
            fn = os.path.join(outdir, spec.filename)
            if skip_existing and os.path.exists(fn):
                continue
            todo.append((spec, fn))
        print(f'Downloading {len(todo)} patches ({len(specs) - len(todo)} already exist)')

        failed = set()
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {pool.submit(self.download_patch, spec, fn): spec for spec, fn in todo}
            for k, fut in enumerate(as_completed(futures)):
                spec = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f'Failed to download patch for sample {spec.sample_id} '
                          f'(slide {spec.slide_id}): {e}', file=sys.stderr)
                    failed.add(spec.sample_id)
                if (k + 1) % 50 == 0 or k + 1 == len(todo):
                    print(f'  {k + 1}/{len(todo)} patches processed')
        return failed


def write_outputs(df, outdir, failed):
    """Write the manifest and specimen list."""
    ok = ~df.id.isin(failed)
    if (~ok).any():
        print(f'Warning: {(~ok).sum()} samples dropped from manifest due to failed downloads')
    df.loc[ok].to_csv(os.path.join(outdir, 'manifest.csv'), index=False)

    with open(os.path.join(outdir, 'specimens.txt'), 'wt') as f:
        for sp in sorted(df.loc[ok].specimen_private.astype(str).unique()):
            f.write(sp + '\n')


def clone_samples_main(args):
    if args.no_verify:
        # The client toggles this warning filter per request, which is not thread-safe
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    conn = Client(args.server, args.apikey, verify=not args.no_verify)
    task = DLTrainingTask(conn, args.task)
    os.makedirs(args.outdir, exist_ok=True)

    cloner = SampleCloner(task, resolution=args.resolution, patch_size=args.patch_size,
                          padding=args.padding, tile_size=args.tile_size)

    df = cloner.load_samples(
        specimens=args.specimens, exclude_specimens=args.exclude_specimens, stains=args.stains,
        labels=args.labels, exclude_labels=args.exclude_labels,
        users=args.users, exclude_users=args.exclude_users,
        max_per_label=args.max_per_label, seed=args.seed,
        raw_manifest_file=os.path.join(args.outdir, 'manifest_server.csv'))
    if len(df) == 0:
        print('No samples to download')
        return

    specs, df = cloner.plan_patches(df)
    npix = sum(s.width * s.height for s in specs)
    print(f'Planned {len(specs)} patches, {npix / 1e6:.1f} Mpixels total')

    failed = set()
    if not args.no_patches:
        failed = cloner.download_all(specs, args.outdir, jobs=args.jobs,
                                     skip_existing=args.skip_existing)

    write_outputs(df, args.outdir, failed)
    print(f'Wrote manifest to {os.path.join(args.outdir, "manifest.csv")}')
