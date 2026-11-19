This repository contains analysis pipelines for the PICSL Histology Annotation Server (PHAS).

Install with `pip install .` (or `pip install -e .` for development), which provides the
`phas-gears` command. Each gear is a sub-command, e.g. `phas-gears clone_samples -h`.
`python -m phas_gears` works as well.

## Gears

### `clone_samples`: download training sample patches

Downloads image patches around the training samples (user-placed boxes) in a PHAS
classifier training (`dltrain`) task, optionally resampled to a target resolution.

```sh
phas-gears clone_samples -s https://histo.itksnap.org -k api_key.json \
    -t 1 -o samples/ -r 0.4 -p 256
```

Patch geometry (sizes and padding are in output pixels, i.e., at the target resolution):

* `-r/--resolution MPP` target resolution in µm/pixel (default: native slide resolution).
  Patches are read from the coarsest pyramid level that is at least as fine as the target
  and resampled.
* `-p/--patch-size N` makes each sample's patch N×N pixels centered on the box; otherwise
  the patch covers the box plus `--padding N` pixels.

Filters (lists can be read from a file with `@file`, one entry per line):

* `--specimens`, `--exclude-specimens`, `--stains`
* `--labels`, `--exclude-labels`
* `--users` (creator *and* editor must be listed), `--exclude-users` (creator *or* editor
  listed), given as numeric PHAS user ids or usernames
* `-l/--max-per-label N` random subset of N samples per label (`--seed`)

Other options: `-P/--no-patches` (only write the manifest),
`--skip-existing` (resume an interrupted download), `-j/--jobs` (parallel downloads).

Outputs in the output folder:

* `manifest_server.csv`: the unfiltered sample manifest from the server
* `manifest.csv`: one row per sample, with server columns plus `slide_id`, `creator_id`,
  `editor_id`, `patch` (path to the PNG), the patch origin and scale in slide pixels (`patch_x0`, `patch_y0`, `patch_scale`), patch size, and the sample's box in
  patch pixel coordinates (`box_x`, `box_y`, `box_w`, `box_h`)
* `specimens.txt`: specimens included
* `patches/`: `<sample_id>.png`
