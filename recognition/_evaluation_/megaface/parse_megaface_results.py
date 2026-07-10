from __future__ import print_function

import argparse
import glob
import json
import os

from megaface_metrics import compute_map, parse_cmc_result


def main():
    parser = argparse.ArgumentParser(
        description='Parse MegaFace cmc_*.json and print standard metrics')
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument('--algo', type=str, default='buffalo_l')
    parser.add_argument('--gallery-size', type=int, default=1000000)
    parser.add_argument('--result-file', type=str, default='',
                        help='optional single cmc json path')
    parser.add_argument('--feature-dir-clean', type=str, default='',
                        help='optional cleaned feature root for mAP')
    parser.add_argument('--facescrub-lst', type=str, default='')
    parser.add_argument('--devkit-root', type=str, default='')
    args = parser.parse_args()

    if args.result_file:
        result_files = [args.result_file]
    else:
        pattern = os.path.join(
            args.result_dir,
            'cmc_facescrub_megaface_%s_%d_*.json' % (args.algo, args.gallery_size))
        result_files = sorted(glob.glob(pattern))

    if not result_files:
        raise SystemExit('No result files found in %s' % args.result_dir)

    print('MegaFace evaluation summary')
    print('algo=%s gallery=%d' % (args.algo, args.gallery_size))
    print('-' * 72)
    for result in result_files:
        with open(result, 'r') as fp:
            data = json.load(fp)
        metrics = parse_cmc_result(data)
        print('file:', os.path.basename(result))
        print('  MegaFace Rank-1 (Id): %.4f%%' % metrics['rank1'])
        if metrics['traditional_rank1'] is not None:
            print('  Traditional Rank-1: %.4f%%' % metrics['traditional_rank1'])
        for far in (1e-4, 1e-5, 1e-6):
            if far in metrics['tar_at_far']:
                item = metrics['tar_at_far'][far]
                print('  TAR@FAR=%g (Ver): %.4f%% (matched FAR=%.2e)' % (
                    far, item['tar'], item['matched_far']))
        print('-' * 72)

    if args.feature_dir_clean and args.facescrub_lst and args.devkit_root:
        template_path = os.path.join(
            args.devkit_root, 'templatelists',
            'megaface_features_list.json_%d_1' % args.gallery_size)
        if not os.path.isfile(template_path):
            raise SystemExit('MegaFace template list not found: %s' % template_path)

        def progress_cb(done, total):
            if done == 1 or done == total or done % 200 == 0:
                print('  mAP progress: %d/%d (%.1f%%)' % (
                    done, total, 100.0 * done / total))

        print('Computing mAP (this may take a while for large galleries)...')
        map_score = compute_map(
            args.feature_dir_clean,
            args.facescrub_lst,
            template_path,
            args.algo,
            args.gallery_size,
            progress_cb=progress_cb)
        if map_score is not None:
            print('  MegaFace mAP (Id): %.4f%%' % map_score)
            print('-' * 72)


if __name__ == '__main__':
    main()
