import argparse

def main():
    # Main module entrypoint. Lists of values can be read from a file using @file syntax,
    # with one argument per line, e.g., --exclude-specimens @excluded.txt
    pp = argparse.ArgumentParser(prog='phas-gears',
                                 description='PICSL Histology Annotation Server Gears (Pipelines)',
                                 fromfile_prefix_chars='@')

    # Sub-commands
    subs = pp.add_subparsers(title='actions')
    subs.required = True
    subs.dest = 'action'

    # Common commands across all subparsers
    pq = argparse.ArgumentParser()
    pq.add_argument('-s', '--server', type=str, help='PHAS server URL', required=True)
    pq.add_argument('-k', '--apikey', type=str, help='JSON file with PHAS API key (also can set via PHAS_AUTH_KEY)')
    pq.add_argument('-v', '--no-verify', action='store_true', help='Disable SSL key verification')


    # Parser for wildcat_inference_on_sampling_rois
    p_wisr = subs.add_parser('wildcat_roi', parents=[pq], add_help=False,
                             description='WildCat inference on sampling ROIs',
                             help='Perform inference on sampling ROIs in a PHAS task')
    p_wisr.add_argument('-t', '--task', type=int, help='PHAS task id', required=True)
    p_wisr.add_argument('-m', '--model', type=str, help='Path to the Wildcat trained model')
    p_wisr.add_argument('--specimens', type=str, nargs='+', help='Limit analysis to a set of specimens')
    p_wisr.add_argument('--stains', type=str, nargs='+', help='Limit analysis to a set of stains')
    p_wisr.add_argument('--authors', type=int, nargs='+', help='Limit analysis to ROIs by certain authors. Provides a list of PHAS user ids. If both creator and editor are in the list, the ROI will be included.')
    p_wisr.add_argument('-o', '--outdir', type=str, help='Path to the output folder', required=True)
    p_wisr.add_argument('-p', '--param', type=argparse.FileType('rt'), required=True, help='Parameter .yaml file')
    p_wisr.add_argument('-n', '--newer', action="store_true", help='No overriding of existing results unless ROIs have changed')

    # Parser for clone_samples
    p_cs = subs.add_parser('clone_samples', parents=[pq], add_help=False,
                           description='Download image patches containing training samples (boxes) '
                                       'from a PHAS classifier training task. Patch sizes and padding '
                                       'are in output pixels, i.e., at the target resolution.',
                           help='Download training sample patches from a PHAS dltrain task')
    p_cs.add_argument('-t', '--task', type=int, help='PHAS task id (classifier training task)', required=True)
    p_cs.add_argument('-o', '--outdir', type=str, help='Path to the output folder', required=True)
    p_cs.add_argument('-r', '--resolution', type=float, metavar='MPP',
                      help='Target resolution in microns per pixel (e.g., 0.4). Default: native slide resolution')
    p_cs.add_argument('-p', '--patch-size', type=int, metavar='N',
                      help='Make each sample\'s patch N x N pixels centered on the box. '
                           'Default: patch covers the box plus padding')
    p_cs.add_argument('--padding', type=int, default=0, metavar='N',
                      help='Padding around the box in pixels, when --patch-size is not used')
    p_cs.add_argument('-S', '--server-patches', action='store_true',
                      help='Fast mode: download the patches already stored on the server for each '
                           'sample (fixed size, typically 512x512 at full resolution, centered on '
                           'the box) instead of extracting new patches from the slides. '
                           'Cannot be combined with -r, -p or --padding')
    p_cs.add_argument('--specimens', type=str, nargs='+', help='Only include these specimens')
    p_cs.add_argument('--exclude-specimens', type=str, nargs='+', help='Exclude these specimens')
    p_cs.add_argument('--stains', type=str, nargs='+', help='Only include these stains')
    p_cs.add_argument('--labels', type=str, nargs='+', help='Only include samples with these label names')
    p_cs.add_argument('--exclude-labels', type=str, nargs='+', help='Exclude samples with these label names')
    p_cs.add_argument('--users', type=str, nargs='+',
                      help='Only include samples whose creator and editor are both in this list (user ids or usernames)')
    p_cs.add_argument('--exclude-users', type=str, nargs='+',
                      help='Exclude samples whose creator or editor is in this list (user ids or usernames)')
    p_cs.add_argument('-l', '--max-per-label', type=int, metavar='N',
                      help='Limit to N randomly selected samples per label')
    p_cs.add_argument('--seed', type=int, default=0, help='Random seed for --max-per-label')
    p_cs.add_argument('-P', '--no-patches', action='store_true',
                      help='Do not download patches, only write the manifest')
    p_cs.add_argument('--skip-existing', action='store_true',
                      help='Do not re-download patches that already exist in the output folder')
    p_cs.add_argument('-j', '--jobs', type=int, default=4, help='Number of parallel downloads (default: 4)')
    p_cs.add_argument('--tile-size', type=int, default=1024, help=argparse.SUPPRESS)

    # Parser for organize_samples (works offline, so it does not take the server options)
    p_os = subs.add_parser('organize_samples',
                           description='Organize patches downloaded by clone_samples into '
                                       'patches/{train,val,test}/<class>/ folders for classifier '
                                       'training. Samples are split so that each specimen/slide is in '
                                       'a single fold, with class proportions kept similar across folds.',
                           help='Split downloaded training samples into train/val/test folders')
    p_os.add_argument('-i', '--input', type=str, required=True, metavar='DIR',
                      help='Output folder of clone_samples (with manifest.csv and patches/)')
    p_os.add_argument('-o', '--outdir', type=str, required=True, metavar='DIR',
                      help='Experiment folder to create; its patches/ subfolder is rebuilt on each run')
    p_os.add_argument('-c', '--classes', type=str, required=True, metavar='JSON',
                      help='JSON file mapping PHAS labels (regular expressions, full match) to classes')
    p_os.add_argument('--split-by', choices=['specimen', 'slide', 'sample'], required=True,
                      help='Unit kept together when splitting into folds. "specimen" gives the most '
                           'honest validation/test estimates; "sample" lets near-duplicate samples '
                           'from the same tissue end up in different folds')
    p_os.add_argument('--val-fraction', type=float, default=0.2, metavar='F',
                      help='Fraction of samples for validation (default: 0.2)')
    p_os.add_argument('--test-fraction', type=float, default=0.0, metavar='F',
                      help='Fraction of samples for testing (default: 0)')
    p_os.add_argument('--val-specimens', type=str, nargs='+',
                      help='Use these specimens for validation (replaces --val-fraction)')
    p_os.add_argument('--test-specimens', type=str, nargs='+',
                      help='Use these specimens for testing (replaces --test-fraction)')
    p_os.add_argument('--exclude-specimens', type=str, nargs='+', help='Leave out these specimens')
    p_os.add_argument('--max-train', type=int, metavar='N', help='Max samples per class in train (default: no limit)')
    p_os.add_argument('--max-val', type=int, metavar='N', help='Max samples per class in val (default: no limit)')
    p_os.add_argument('--max-test', type=int, metavar='N', help='Max samples per class in test (default: no limit)')
    p_os.add_argument('--balance-val', action='store_true',
                      help='Downsample validation so that all classes have as many samples as the smallest')
    p_os.add_argument('--drop-unmatched', action='store_true',
                      help='Drop samples whose label matches no class (default: report an error)')
    p_os.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')

    # Parse the arguments
    args = pp.parse_args()

    # With SSL verification disabled, do not print a warning on every request. The PHAS client
    # resets warning filters around each request (which overrides urllib3.disable_warnings and
    # is not thread-safe), so the warning is dropped at display time instead.
    if getattr(args, 'no_verify', False):
        import warnings
        from urllib3.exceptions import InsecureRequestWarning
        _showwarning = warnings.showwarning
        def _showwarning_filtered(message, category, *a, **kw):
            if not issubclass(category, InsecureRequestWarning):
                _showwarning(message, category, *a, **kw)
        warnings.showwarning = _showwarning_filtered

    # Execute the correct command. Imports are deferred so that each gear only loads its own dependencies
    if args.action == 'wildcat_roi':
        from .wildcat_inference_on_sampling_rois import wildcat_inference_on_sampling_rois_main
        print(args)
        wildcat_inference_on_sampling_rois_main(args)
    elif args.action == 'clone_samples':
        if args.server_patches and (args.resolution or args.patch_size or args.padding):
            p_cs.error('--server-patches cannot be combined with --resolution, --patch-size or --padding')
        from .clone_samples import clone_samples_main
        clone_samples_main(args)
    elif args.action == 'organize_samples':
        from .organize_samples import organize_samples_main
        organize_samples_main(args)


if __name__ == '__main__':
    main()
