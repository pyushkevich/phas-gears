This repository contains analysis pipelines ("gears") for the PICSL Histology Annotation
Server (PHAS).

## Installation and usage

Install with `pip install .` (or `pip install -e .` for development), which provides the
`phas-gears` command. Each gear is a sub-command; run `phas-gears <gear> -h` for its full list
of options. `python -m phas_gears` works as well.

Gears that talk to the server share these options:

* `-s/--server URL` PHAS server, e.g. `https://histo.itksnap.org`
* `-k/--apikey FILE` JSON file with your PHAS API key (generate it from the web interface);
  alternatively set the `PHAS_AUTH_KEY` environment variable to the file's path
* `-v/--no-verify` disable SSL certificate verification

Options that take a list (e.g. `--specimens`) can read it from a file with `@file`, one entry
per line, e.g. `--exclude-specimens @excluded.txt`.

A typical classifier training workflow is:

1. `clone_samples` downloads patches around the training boxes of a `dltrain` task
2. `organize_samples` splits them into train/val/test folders for training
3. train with `tangle_cnn`, then apply the model to sampling ROIs with `wildcat_roi`

## Gears

### `clone_samples`: download training sample patches

Downloads image patches around the training samples (user-placed boxes) in a PHAS
classifier training (`dltrain`) task, optionally resampled to a target resolution.

```sh
phas-gears clone_samples -s https://histo.itksnap.org -k api_key.json \
    -t 1 -o samples/ -r 0.4 -p 256
```

**Fast option:** `-S/--server-patches` downloads the patches that the server already stores
for each sample, which is much faster than extracting new patches from the slides (one
small request per sample, no resampling). It also works for slides whose image files are no
longer available on the server. Use it when the stored patches suit your needs:

```sh
phas-gears clone_samples -s https://histo.itksnap.org -k api_key.json -t 1 -o samples/ -S
```

Stored patches have a fixed size set by the task (`display-patch-size`, default 512×512),
at full slide resolution, centered on the box center rounded to a whole pixel. They cannot
be combined with `-r`, `-p` or `--padding`. A few caveats inherited from the server: patches
of samples edited after creation are regenerated at 512×512 regardless of the task setting,
and samples created by some server-side import tools may have patches from a lower
resolution level (the manifest assumes full resolution). Patches generated before September
2019 are positioned one pixel differently from newer ones; the manifest accounts for this
based on each sample's last edit time.

Otherwise, patches are extracted from the slide with the following geometry (sizes and
padding are in output pixels, i.e., at the target resolution):

* `-r/--resolution MPP` target resolution in µm/pixel (default: native slide resolution).
  Patches are read from the coarsest pyramid level that is at least as fine as the target
  and resampled.
* `-p/--patch-size N` makes each sample's patch N×N pixels centered on the box; otherwise
  the patch covers the box plus `--padding N` pixels on each side.

Samples on slides whose image files are not available on the server are skipped with a
warning (use `-S` to get their stored patches).

Filters:

* `--specimens`, `--exclude-specimens`, `--stains`
* `--labels`, `--exclude-labels` (label names)
* `--users` (creator *and* editor must be listed), `--exclude-users` (creator *or* editor
  listed), given as numeric PHAS user ids or usernames
* `-l/--max-per-label N` random subset of N samples per label (`--seed`)

Other options: `-P/--no-patches` (only write the manifest), `--skip-existing` (resume an
interrupted download), `-j/--jobs N` (parallel downloads, default 4).

Outputs in the output folder:

* `patches/<sample_id>.png`: the patches
* `manifest.csv`: one row per sample, with the server's columns plus
  * `slide_id`, `creator_id`, `editor_id`: numeric ids
  * `patch`: path of the patch, relative to the output folder
  * `patch_x0`, `patch_y0`, `patch_scale`: patch origin in slide pixels and size of a patch
    pixel in slide pixels; `patch_width`, `patch_height`: patch size; `resolution`
  * `box_x`, `box_y`, `box_w`, `box_h`: the sample's box in patch pixel coordinates

  With `-S -P` the patch geometry columns are empty, since the sizes of stored patches are
  only known once they are downloaded.
* `manifest_server.csv`: the unfiltered sample manifest from the server
* `specimens.txt`: specimens included

### `organize_samples`: split samples into training folds

Organizes the patches downloaded by `clone_samples` into the folder structure used by
`torchvision.datasets.ImageFolder` and `tangle_cnn` training. It works offline (no server
options):

```sh
phas-gears organize_samples -i samples/ -o exp01/ -c classes.json --split-by specimen \
    --val-fraction 0.2 --test-fraction 0.2 --balance-val
```

Classes are defined by a JSON file. Each label pattern is a regular expression that must match
the whole PHAS label name (so `tissue` does not match `tissue_fold`). A label belongs to the
first class that matches it, and classes with `"ignore": 1` are left out. Labels that match
no class are an error unless `--drop-unmatched` is given. The gear prints which class each
label was assigned to.

```json
[
  { "classname": "tangle", "labels": [ "tangle", "pretangle" ] },
  { "classname": "ignore", "labels": [ "ask_dave", "ambiguous" ], "ignore": 1 },
  { "classname": "nontangle", "labels": [ ".*" ] }
]
```

Splitting:

* `--split-by specimen|slide|sample` (required) sets the unit that is kept together in one
  fold. Splitting by `specimen` gives the most honest validation and test estimates; with
  `sample`, neighboring or duplicate boxes from the same tissue can end up in different
  folds, inflating validation accuracy (which `tangle_cnn` uses to select the best epoch).
  With few specimens, `slide` is a compromise.
* `--val-fraction` (default 0.2) and `--test-fraction` (default 0) set the fold sizes. Units
  are assigned so that each fold gets close to the requested fraction of every class; the
  achievable precision depends on how many units there are.
* `--val-specimens` / `--test-specimens` put specific specimens in a fold instead.
* `--exclude-specimens` leaves specimens out; `--seed` makes the split reproducible.

Balancing: `tangle_cnn` already balances classes during training (weighted sampling), so
limiting training samples is usually unnecessary. Validation accuracy, however, is dominated
by the largest classes, so `--balance-val` downsamples validation to the same number of
samples per class. `--max-train`, `--max-val` and `--max-test` cap samples per class.

Every class must have samples in both train and val, otherwise `ImageFolder` would number the
classes differently in the two folds; the gear stops with an error if this is not the case.

Outputs in the experiment folder:

* `patches/{train,val,test}/<class>/<sample_id>.png`: links to the downloaded patches. This
  folder is deleted and rebuilt on every run (the gear refuses if it contains regular files).
* `split_manifest.csv`: the samples with their `class` and `fold`. Pass this file as the
  manifest to `tangle_cnn` training. When the patches were resampled (`clone_samples -r`),
  `w`, `h`, `mpp_x` and `mpp_y` are rewritten to describe the saved patch (box size in patch
  pixels, patch resolution in mm), as `tangle_cnn` expects; the original values are kept in
  `*_slide` columns.
* `split_summary.csv`, plus `classes.json` and `organize_args.json` to reproduce the split.

### `wildcat_roi`: WildCat inference on sampling ROIs

Applies a trained WildCat model to the sampling ROIs in a PHAS sampling task and computes
per-ROI statistics (density maps, segmentations and summary tables).

```sh
phas-gears wildcat_roi -s https://histo.itksnap.org -k api_key.json \
    -t 39 -m models/my_model -p inference_param.yml -o results/
```

* `-m/--model DIR` trained WildCat model; `-p/--param FILE` inference parameter `.yml` file
* `--specimens`, `--stains` limit the analysis to these specimens/stains
* `--authors` limit to ROIs whose creator and editor are both in this list of PHAS user ids
* `-n/--newer` skip slides whose ROIs have not changed since the last run
