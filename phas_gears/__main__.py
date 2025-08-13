import argparse
from .wildcat_inference_on_sampling_rois import wildcat_inference_on_sampling_rois_main

# Main module entrypoint
pp = argparse.ArgumentParser(description='PICSL Histology Annotation Server Gears (Pipelines)')

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
p_wisr.add_argument('-o', '--outdir', type=str, help='Path to the output folder', required=True)
p_wisr.add_argument('-p', '--param', type=argparse.FileType('rt'), required=True, help='Parameter .yaml file')
p_wisr.add_argument('-n', '--newer', action="store_true", help='No overriding of existing results unless ROIs have changed')

# Parse the arguments
args = pp.parse_args()

# Execute the correct command
if args.action == 'wildcat_roi':
    print(args)
    wildcat_inference_on_sampling_rois_main(args)