# Python 3 port of MegaFace devkit run_experiment.py
# Original: http://megaface.cs.washington.edu/

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys


def setup_devkit_runtime(devkit_root):
  lib_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'third_party', 'lib')
  if os.path.isdir(lib_dir):
    prev = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = (
        lib_dir + (':' + prev if prev else ''))


def main():
    parser = argparse.ArgumentParser(
        description='Run MegaFace identification experiment (Python 3)')
    parser.add_argument('--devkit-root', type=str, required=True,
                        help='path to megaface devkit directory')
    parser.add_argument('distractor_feature_path',
                        help='path to MegaFace feature root')
    parser.add_argument('probe_feature_path',
                        help='path to FaceScrub feature root')
    parser.add_argument('file_ending',
                        help='feature suffix, e.g. _buffalo_l.bin')
    parser.add_argument('out_root', help='directory for result json files')
    parser.add_argument('-s', '--sizes', type=int, nargs='+',
                        default=[1000000],
                        help='distractor set sizes, default: 1000000')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='scoring model, default: devkit/models/jb_identity.bin')
    parser.add_argument('-ns', '--num-sets', type=int, default=1)
    parser.add_argument('-d', '--delete-matrices', action='store_true')
    parser.add_argument('-p', '--probe-list', type=str, default=None)
    parser.add_argument('-dlp', '--distractor-list-path', type=str, default=None)
    args = parser.parse_args()

    devkit_root = os.path.abspath(args.devkit_root)
    setup_devkit_runtime(devkit_root)
    model = args.model or os.path.join(devkit_root, 'models', 'jb_identity.bin')
    identification_exe = os.path.join(devkit_root, 'bin', 'Identification')
    fuse_results_exe = os.path.join(devkit_root, 'bin', 'FuseResults')
    megaface_list_basename = os.path.join(
        devkit_root, 'templatelists', 'megaface_features_list.json')
    probe_list_basename = args.probe_list or os.path.join(
        devkit_root, 'templatelists', 'facescrub_features_list.json')
    distractor_list_path = args.distractor_list_path or os.path.join(
        devkit_root, 'templatelists')

    distractor_feature_path = os.path.abspath(args.distractor_feature_path)
    probe_feature_path = os.path.abspath(args.probe_feature_path)
    out_root = os.path.abspath(args.out_root)
    file_ending = args.file_ending
    alg_name = file_ending.split('.')[0].strip('_')
    sizes = args.sizes
    set_indices = range(1, int(args.num_sets) + 1)

    print('distractor features:', distractor_feature_path)
    print('probe features:', probe_feature_path)
    print('file ending:', file_ending)
    print('scoring model:', model)

    assert os.path.exists(distractor_feature_path), distractor_feature_path
    assert os.path.exists(probe_feature_path), probe_feature_path
    assert os.path.isfile(identification_exe), identification_exe
    assert os.path.isfile(fuse_results_exe), fuse_results_exe

    os.makedirs(out_root, exist_ok=True)
    other_out_root = os.path.join(out_root, 'otherFiles')
    os.makedirs(other_out_root, exist_ok=True)

    probe_name = os.path.basename(probe_list_basename).split('_')[0]
    distractor_name = os.path.basename(megaface_list_basename).split('_')[0]
    megaface_list_basename = os.path.join(
        distractor_list_path, os.path.basename(megaface_list_basename))

    missing = False
    for index in set_indices:
        for size in sizes:
            print('Creating feature list of %d photos for set %d' % (size, index))
            cur_list_name = '%s_%d_%d' % (megaface_list_basename, size, index)
            with open(cur_list_name) as fp:
                feature_file = json.load(fp)
                path_list = feature_file['path']
                for i, rel_path in enumerate(path_list):
                    path_list[i] = os.path.join(
                        distractor_feature_path, rel_path + file_ending)
                    if not os.path.isfile(path_list[i]):
                        print('%s is missing' % path_list[i])
                        missing = True
                    if i % 10000 == 0 and i > 0:
                        print('%d / %d' % (i, len(path_list)))
                feature_file['path'] = path_list
                out_name = os.path.join(
                    other_out_root,
                    '%s_features_%s_%d_%d' % (distractor_name, alg_name, size, index))
                with open(out_name, 'w') as out_fp:
                    json.dump(feature_file, out_fp, sort_keys=True, indent=4)

    if missing:
        sys.exit('Features are missing...')

    with open(probe_list_basename) as fp:
        feature_file = json.load(fp)
        path_list = feature_file['path']
        for i, rel_path in enumerate(path_list):
            path_list[i] = os.path.join(probe_feature_path, rel_path + file_ending)
            if not os.path.isfile(path_list[i]):
                print('%s is missing' % path_list[i])
                missing = True
        feature_file['path'] = path_list
        probe_feature_list = os.path.join(
            other_out_root, '%s_features_%s' % (probe_name, alg_name))
        with open(probe_feature_list, 'w') as out_fp:
            json.dump(feature_file, out_fp, sort_keys=True, indent=4)

    if missing:
        sys.exit('Features are missing...')

    print('Running probe to probe comparison')
    probe_score_filename = os.path.join(
        other_out_root, '%s_%s_%s.bin' % (probe_name, probe_name, alg_name))
    subprocess.check_call([
        identification_exe, model, 'path', probe_feature_list,
        probe_feature_list, probe_score_filename])

    for index in set_indices:
        for size in sizes:
            print('Running test with size %d images for set %d' % (size, index))
            distractor_feature_list = os.path.join(
                other_out_root,
                '%s_features_%s_%d_%d' % (distractor_name, alg_name, size, index))
            score_matrix = os.path.join(
                other_out_root,
                '%s_%s_%s_%d_%d.bin' % (
                    probe_name, distractor_name, alg_name, size, index))
            subprocess.check_call([
                identification_exe, model, 'path', distractor_feature_list,
                probe_feature_list, score_matrix])

            print('Computing test results with %d images for set %d' % (size, index))
            cmc_json = os.path.join(
                out_root,
                'cmc_%s_%s_%s_%d_%d.json' % (
                    probe_name, distractor_name, alg_name, size, index))
            matches_json = os.path.join(
                out_root,
                'matches_%s_%s_%s_%d_%d.json' % (
                    probe_name, distractor_name, alg_name, size, index))
            subprocess.check_call([
                fuse_results_exe, score_matrix, probe_score_filename,
                probe_feature_list, str(size), cmc_json, matches_json])

            if args.delete_matrices and os.path.isfile(score_matrix):
                os.remove(score_matrix)


if __name__ == '__main__':
    main()
