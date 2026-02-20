import sys
import traceback
import re
import deeplabcut
import os
from procrustean_analysis import get_angles_procrustes
import pandas as pd

# Usage: python dlc_worker.py <config_path> <video_path> <output_dir>

def main():
    # Batch mode: python dlc_worker.py <config_path> <video1> <video2> ... --batch <save_as_csv> <pcutoff> <overwrite>
    if '--batch' in sys.argv:
        batch_idx = sys.argv.index('--batch')
        config_path = sys.argv[1]
        video_paths = sys.argv[2:batch_idx]
        save_as_csv = sys.argv[batch_idx+1].lower() == 'true'
        get_angles = sys.argv[batch_idx+2].lower() == 'true'
        pcutoff = float(sys.argv[batch_idx+3]) if len(sys.argv) > batch_idx+3 else 0.6
        overwrite = sys.argv[batch_idx+4].lower() == 'true' if len(sys.argv) > batch_idx+4 else False
        failed = []
        try:
            print(f"[DLC Worker] Batch analyzing {len(video_paths)} videos")
            deeplabcut.analyze_videos(
                config_path,
                video_paths,
                save_as_csv=save_as_csv,
                batchsize=1,
                destfolder=os.path.dirname(video_paths[0]) if video_paths else None,
                use_shelve=True
            )
        except Exception as e:
            print(f"[DLC Worker] ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
            # Mark all as failed if batch call fails
            failed = video_paths
        # Check for output files to determine which succeeded
        for v in video_paths:
            base, ext = os.path.splitext(v)
            # Find any file in the directory that starts with base + 'DLC_Resnet'
            dirpath = os.path.dirname(v)
            prefix = os.path.basename(base) + 'DLC_Resnet'
            # check if there is a file that starts with the prefix in dirpath
            fns = os.listdir(dirpath)
            found = any([fn for fn in fns if fn.startswith(prefix) and fn.endswith('.h5')])
            if not found:
                print(f"FAILED: {v}", file=sys.stderr)
                failed.append(v)
        if failed:
            sys.exit(2)
        else:
            sys.exit(0)
    # Single file mode (legacy)
    if len(sys.argv) < 7:
        # print("Usage: python dlc_worker.py <config_path> <video_path> <output_dir> <save_as_csv> <get_angles> <make_labeled_video> <pcutoff> <overwrite>", file=sys.stderr)
        # example usage: python dlc_worker.py ./SuperFly-Pablo-2025-07-29/config.yaml C:\Users\johnp\OneDrive\Desktop\smellovision\old\healthy_fly_exp\demo\2025_09_11_16_12_21_01.mp4 C:\Users\johnp\OneDrive\Desktop\smellovision\old\healthy_fly_exp\demo\ False True False .6 True
        print("Usage: python dlc_worker.py <config_path> <video_path> <output_dir> <save_as_csv> <get_angles> <make_labeled_video> <pcutoff> <overwrite>")
        sys.exit(1)
    config_path = sys.argv[1]
    video_path = sys.argv[2]
    output_dir = sys.argv[3]
    save_as_csv = sys.argv[4].lower() == 'true'
    get_angles = sys.argv[5].lower() == 'true'
    make_labeled_video = sys.argv[6].lower() == 'true'
    pcutoff = float(sys.argv[7]) if len(sys.argv) > 7 else 0.6
    overwrite = sys.argv[8].lower() == 'true' if len(sys.argv) > 8 else False
    try:
        print(f"[DLC Worker] Starting analysis for {video_path}")
        # check if there's already an output for this file by looking for files with the same basename that end in .h5
        output_fns = [fn for fn in os.listdir(output_dir) if fn.endswith('.h5')]
        basename = os.path.basename(video_path).split(".")[0]
        old_fn = None
        other_fns = []
        for fn in output_fns:
            if basename in fn and 'DLC_Resnet' in fn:
                old_fn = fn
                # and find others that contain THIS basename
                old_basename = os.path.basename(old_fn).split(".")[0]
                for fn in os.listdir(output_dir):
                    if old_basename in fn and fn != old_fn:
                        other_fns.append(fn)
                break
        # if so, delete it
        if overwrite and old_fn is not None:
            os.remove(os.path.join(output_dir, old_fn))
            for fn in other_fns:
                os.remove(os.path.join(output_dir, fn))
        # analyze
        deeplabcut.analyze_videos(
            config_path,
            [video_path],
            videotype=os.path.splitext(video_path)[1],
            destfolder=output_dir,
            save_as_csv=save_as_csv,
            batchsize=1,
        )
        print(f"[DLC Worker] Analysis complete for {video_path}")
        if make_labeled_video:
            print(f"[DLC Worker] Creating labeled video for {video_path}")
            deeplabcut.create_labeled_video(
                config_path,
                [video_path],
                videotype=os.path.splitext(video_path)[1],
                pcutoff=pcutoff,
                overwrite=overwrite,
                fastmode=True
            )
            print(f"[DLC Worker] Labeled video created for {video_path}")
        if get_angles:
            output_fns = [fn for fn in os.listdir(output_dir) if fn.endswith('.h5')]
            basename = os.path.basename(video_path).split(".")[0]
            old_fn = None
            for fn in output_fns:
                if basename in fn and 'DLC_Resnet' in fn:
                    old_fn = fn
                    break
            if old_fn is not None:
                h5_file = os.path.join(output_dir, old_fn)
                df = pd.read_hdf(h5_file)
                # the dataframe columns have 3 levels. get subset by the first
                first_key = df.keys()[0][0]
                centers, angles = get_angles_procrustes(df[first_key])
                # add the body angles to the dataframe
                df['body_angle'] = angles
                df['body_x'] = centers[:, 0]
                df['body_y'] = centers[:, 1]
                # and save
                df.to_hdf(h5_file, key='df', mode='w')
                print(f"[DLC Worker] Angles for {video_path}: {angles}")
        sys.exit(0)
    except Exception as e:
        print(f"[DLC Worker] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
