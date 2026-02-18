import sys
import traceback
import re
import deeplabcut
import os

# Usage: python dlc_worker.py <config_path> <video_path> <output_dir>

def main():
    # Batch mode: python dlc_worker.py <config_path> <video1> <video2> ... --batch <save_as_csv> <pcutoff> <overwrite>
    if '--batch' in sys.argv:
        batch_idx = sys.argv.index('--batch')
        config_path = sys.argv[1]
        video_paths = sys.argv[2:batch_idx]
        save_as_csv = sys.argv[batch_idx+1].lower() == 'true'
        pcutoff = float(sys.argv[batch_idx+2]) if len(sys.argv) > batch_idx+2 else 0.6
        overwrite = sys.argv[batch_idx+3].lower() == 'true' if len(sys.argv) > batch_idx+3 else False
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
    if len(sys.argv) < 6:
        # print("Usage: python dlc_worker.py <config_path> <video_path> <output_dir> <save_as_csv> <make_labeled_video> <pcutoff> <overwrite>", file=sys.stderr)
        print("Usage: python dlc_worker.py <config_path> <video_path> <output_dir> <save_as_csv> <make_labeled_video> <pcutoff> <overwrite>")
        sys.exit(1)
    config_path = sys.argv[1]
    video_path = sys.argv[2]
    output_dir = sys.argv[3]
    save_as_csv = sys.argv[4].lower() == 'true'
    make_labeled_video = sys.argv[5].lower() == 'true'
    pcutoff = float(sys.argv[6]) if len(sys.argv) > 6 else 0.6
    overwrite = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False
    try:
        print(f"[DLC Worker] Starting analysis for {video_path}")
        deeplabcut.analyze_videos(
            config_path,
            [video_path],
            videotype=os.path.splitext(video_path)[1],
            destfolder=output_dir,
            save_as_csv=save_as_csv,
            batchsize=1,
            use_shelve=True
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
        sys.exit(0)
    except Exception as e:
        print(f"[DLC Worker] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
