import os
import glob
import re
import pandas as pd

class Video:
    def __init__(self, path):
        self.path = path

    def is_extracted(self):
        """
        Checks if frames have been extracted for this video.
        Assumes labeled frames are in a directory named 'labeled-data/<video_basename>'.
        """
        base = os.path.splitext(os.path.basename(self.path))[0]
        self.labeled_dir = os.path.join(os.path.dirname(os.path.dirname(self.path)), 'labeled-data', base)
        if not os.path.isdir(self.labeled_dir):
            return False
        else:
            # check if there are at least 10 extracted frames
            self.extracted_frames = glob.glob(os.path.join(self.labeled_dir, 'img*.png'))
            return len(self.extracted_frames) > 10

    def is_labeled(self):
        """
        Checks if every extracted frame of this video has been labeled.
        Assumes labeled frames are in a directory named 'labeled-data/<video_basename>'.
        """
        ret = False
        if self.is_extracted():
            # check if all of the extracted frames were labeled in the appropriate .csv
            # get all *.h5 files in the labeled directory
            h5_files = glob.glob(os.path.join(self.labeled_dir, '*.h5'))
            # reduce to just the files that say CollectedData
            # h5_files = [f for f in h5_files if 'CollectedData' in os.path.basename(f)]
            # and use just the most recent one
            h5_files.sort(key=os.path.getmtime)
            if len(h5_files) > 0:
                recent_fn = h5_files[-1]
                labeled_frames = []
                for h5_file in h5_files:
                    df = pd.read_hdf(h5_file)
                    # go one level down in the df
                    key1 = df.keys()[0][0]
                    # go row-by-row and store filenames that have label coordinates that are meaningful
                    for lbl, vals in df.iterrows():
                        if not vals[key1].isnull().all():
                            # get the full path of the filename
                            fn = lbl[-1]
                            fn = os.path.join(self.labeled_dir, fn)
                            labeled_frames += [fn]
                # now, find all of the frame files
                for frame in self.extracted_frames:
                    if frame not in labeled_frames:
                        return False
                ret = True
        return ret

    def get_model_num(self):
        """
        Looks up the latest model number trained on this video by searching for DLC result files.
        Assumes result files are named like <video_basename>DLC_Resnet50_*_snapshot_best-XX.h5
        Returns the highest XX found, or None if not found.
        """
        base = os.path.splitext(os.path.basename(self.path))[0]
        pattern = os.path.join(os.path.dirname(self.path), f'{base}DLC_Resnet50_*_snapshot_best-*.h5')
        files = glob.glob(pattern)
        model_nums = []
        for f in files:
            m = re.search(r'snapshot_best-(\d+)', f)
            if m:
                model_nums.append(int(m.group(1)))
        return max(model_nums) if model_nums else None
