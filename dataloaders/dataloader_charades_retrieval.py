from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import csv
import json
import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from dataloaders.rawvideo_util import RawVideoExtractor

class Charades_DataLoader(Dataset):
    """Charades dataset loader."""

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.anno_path = anno_path
        self.video_path = video_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.anno_path, "Charades_v1_train.csv")
        video_id_path_dict["test"] = os.path.join(self.anno_path, "Charades_v1_test.csv")
        cap_json_path = os.path.join(self.anno_path, "Charades_VILA_F6.json")
        with open(cap_json_path, 'r') as f:
            cap_json_data = json.load(f)

        self.all_train_pairs = []
        with open(video_id_path_dict["train"]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                id, descriptions = row["id"], row["descriptions"]
                for cap_row in cap_json_data:
                    if cap_row['name'] == id:
                        seq_cap = cap_row['description']
                        seq_cap = seq_cap.split(".")
                        seq_cap.extend([seq_cap[-1]] * (6 - len(seq_cap)))
                        seq_cap = seq_cap[:6]
                        self.all_train_pairs.append([id, descriptions, seq_cap])
                        break
        print("train len is", len(self.all_train_pairs))

        self.sample_len = len(self.all_train_pairs)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros(self.max_frames, dtype=np.int64)
        max_video_length = 0

        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.video_path, video_id + '.mp4')
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length = max_video_length if max_video_length > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def __getitem__(self, idx):

        if self.subset == "train":
            vid, caption, seq_cap = self.all_train_pairs[idx]
            pairs_text_list, pairs_mask_list, pairs_segment_list = [], [], []
            for cap in seq_cap:
                pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(vid, cap)
                pairs_text_list.append(pairs_text)
                pairs_mask_list.append(pairs_mask)
                pairs_segment_list.append(pairs_segment)
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(vid, caption)
            video, video_mask = self._get_rawvideo(choice_video_ids)
            return pairs_text, pairs_mask, pairs_text_list, pairs_mask_list, video, video_mask, idx, hash(idx)

class Charades_TestDataLoader(Dataset):
    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.anno_path = anno_path
        self.video_path = video_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.anno_path, "Charades_v1_train.csv")
        video_id_path_dict["test"] = os.path.join(self.anno_path, "Charades_v1_test.csv")
        cap_json_path = os.path.join(self.anno_path, "Charades_VILA_F6.json")
        with open(cap_json_path, 'r') as f:
            cap_json_data = json.load(f)

        self.all_test_pairs = []
        with open(video_id_path_dict["test"]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                id, descriptions = row["id"], row["descriptions"]
                for cap_row in cap_json_data:
                    if cap_row['name'] == id:
                        seq_cap = cap_row['description']
                        seq_cap = seq_cap.split(".")
                        seq_cap.extend([seq_cap[-1]] * (6 - len(seq_cap)))
                        seq_cap = seq_cap[:6]
                        self.all_test_pairs.append([id, descriptions, seq_cap])
                        break
        print("test len is", len(self.all_test_pairs))

        self.sample_len = len(self.all_test_pairs)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros(self.max_frames, dtype=np.int64)
        max_video_length = 0

        video = np.zeros(
            (len(choice_video_ids), self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.video_path, video_id + '.mp4')
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length = max_video_length if max_video_length > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def __getitem__(self, idx):

        vid, caption, seq_cap = self.all_test_pairs[idx]
        pairs_text_list, pairs_mask_list, pairs_segment_list = [], [], []
        for cap in seq_cap:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(vid, cap)
            pairs_text_list.append(pairs_text)
            pairs_mask_list.append(pairs_mask)
            pairs_segment_list.append(pairs_segment)
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(vid, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_text_list, pairs_mask_list, video, video_mask, idx, hash(idx)