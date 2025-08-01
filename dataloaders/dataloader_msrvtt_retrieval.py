from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import tempfile
import pandas as pd
from os.path import join, splitext, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset

class MSRVTTDataset(RetrievalDataset):
    """MSRVTT dataset."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(MSRVTTDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words, max_frames, video_framerate, image_resolution, mode, config=config)
        pass

    def _get_anns(self, subset='train'):
        """
        video_dict: dict: video_id -> video_path
        sentences_dict: list: [(video_id, caption)] , caption (list: [text:, start, end])
        """
        csv_path = {'train': join(self.anno_path, 'MSRVTT_train.9000.csv'),
                    'test': join(self.anno_path, 'MSRVTT_test.1000.csv')}[subset]
        if exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError

        video_id_list = list(csv['video_id'].values)

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()
        if subset == 'train':
            anno_path = join(self.anno_path, 'MSRVTT_data.json')
            data = json.load(open(anno_path, 'r'))
            cap_anno_path = join(self.anno_path, 'MSRVTT_VILA_F6.json')
            cap_data = json.load(open(cap_anno_path, 'r'))
            for itm in data['sentences']:
                if itm['video_id'] in video_id_list:
                    for cap_itm in cap_data:
                        if cap_itm['name'] == itm['video_id']:
                            seq_cap = cap_itm['description']
                            seq_cap = seq_cap.split(".")
                            seq_cap.extend([seq_cap[-1]] * (6 - len(seq_cap)))
                            seq_cap = seq_cap[:6]
                            sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['caption'], None, None, seq_cap))
                            video_dict[itm['video_id']] = join(self.video_path, "{}.mp4".format(itm['video_id']))
                            break
        else:
            cap_anno_path = join(self.anno_path, 'MSRVTT_VILA_F6.json')
            cap_data = json.load(open(cap_anno_path, 'r'))
            for _, itm in csv.iterrows():
                for cap_itm in cap_data:
                    if cap_itm['name'] == itm['video_id']:
                        seq_cap = cap_itm['description']
                        seq_cap = seq_cap.split(".")
                        seq_cap.extend([seq_cap[-1]] * (6 - len(seq_cap)))
                        seq_cap = seq_cap[:6]
                        sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['sentence'], None, None, seq_cap))
                        video_dict[itm['video_id']] = join(self.video_path, "{}.mp4".format(itm['video_id']))
                        break

        return video_dict, sentences_dict