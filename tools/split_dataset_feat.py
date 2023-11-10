# ====================================================
# @Time    : 6/5/21 1:32 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : split_dataset_feat.py
# ====================================================
import h5py
import numpy as np
import os
import os.path as osp
import pandas as pd


def np2h5(in_dir, out_dir, video_list, mode):
    out_file = osp.join(out_dir, 'region_16c20b_{}.h5'.format(mode))
    video_fd = h5py.File(out_file, 'w')
    feat_dset, bbox_dset, ids_dset = None, None, None
    bbox_num = 20
    for video in video_list:
        frames_list = os.path.listdir()
        roi_feats = []
        roi_bboxes = []
        for frame in frames_list:
            bbox_file = osp.join(osp.join(in_dir, str(video)), frame + '.npz')
            npz = np.load(bbox_file)
            roi_feat = npz['x']
            # bnum = roi_feat.shape[2]
            roi_bbox = npz['bbox']
            # if bnum < bbox_num:
            #     add_num = bbox_num - bnum
            #     print(add_num)
            #     add_feat, add_bbox = [], []
            #     for _ in range(add_num):
            #         add_feat.append(roi_feat[:, :, bnum-1, :])
            #         add_bbox.append(roi_bbox[:, :, bnum-1, :])
            #     add_feat = np.asarray(add_feat).transpose(1, 2, 0, 3)
            #     add_bbox = np.asarray(add_bbox).transpose(1, 2, 0, 3)
            #     print(add_feat.shape, add_bbox.shape)
            #     roi_feat = np.concatenate((roi_feat, add_feat), axis=2)
            #     roi_bbox = np.concatenate((roi_bbox, add_bbox), axis=2)

            roi_feat = roi_feat[:bbox_num, :]
            roi_bbox = roi_bbox[:bbox_num, :]

            roi_feats.append(roi_feat)
            roi_bboxes.append(roi_bbox)

        nclip, nframe = 8, 4
        clip_feats = np.array(roi_feats).squeeze()
        clip_bboxes = np.array(roi_bboxes).squeeze()
        print("clip_feats: ", clip_feats.shape)
        clip_feats = clip_feats.reshape(nclip, nframe, -1)
        clip_bboxes = clip_bboxes.reshape(nclip, nframe, -1)

        # print(roi_feat.shape, roi_bbox.shape)
        if feat_dset is None:
            dataset_size = len(video_list)
            C, F, R, D = clip_feats.shape
            feat_dset = video_fd.create_dataset('feat', (dataset_size, C, F, R, D),
                                                  dtype=np.float32)
            ids_dset = video_fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
            C, F, R, D = clip_bboxes.shape
            bbox_dset = video_fd.create_dataset('bbox', shape=(dataset_size, C, F, R, D),
                                                dtype=np.float32)
            ival = 0

        feat_dset[ival:(ival + 1)] = clip_feats
        bbox_dset[ival:(ival + 1)] = clip_bboxes
        ids_dset[ival:(ival + 1)] = int(video)

        ival += 1
    print('Save to {}'.format(out_file))

def split_dataset_feat(filename, out_dir, train_list, val_list, test_list):

    train_fd = h5py.File(osp.join(out_dir, 'app_feat_train.h5'), 'w')
    val_fd = h5py.File(osp.join(out_dir, 'app_feat_val.h5'), 'w')
    test_fd = h5py.File(osp.join(out_dir, 'app_feat_test.h5'), 'w')
    val_feat_dset, val_ids_dset = None, None
    test_feat_dset, test_ids_dset = None, None
    train_feat_dset, train_ids_dset = None, None

    feat_name = 'resnet_features'
    with h5py.File(filename, 'r') as fp:
        vids = fp['ids']
        feats = fp[feat_name]
        for vid, feat in zip(vids, feats):
            if vid in val_list:
                if val_feat_dset is None:
                    dataset_size = len(val_list)
                    C, F, D = feat.shape
                    # C, D = feat.shape
                    val_feat_dset = val_fd.create_dataset(feat_name, (dataset_size, C, F, D),
                                                      dtype=np.float32)
                    val_ids_dset = val_fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
                    ival = 0
                val_feat_dset[ival:(ival+1)] = feat
                val_ids_dset[ival:(ival+1)] = int(vid)
                ival += 1
            elif vid in test_list:
                if test_feat_dset is None:
                    dataset_size = len(test_list)
                    C, F, D = feat.shape
                    # C, D = feat.shape
                    test_feat_dset = test_fd.create_dataset(feat_name, (dataset_size, C, F, D),
                                                      dtype=np.float32)
                    test_ids_dset = test_fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
                    itest = 0

                test_feat_dset[itest:(itest + 1)] = feat
                test_ids_dset[itest:(itest + 1)] = int(vid)
                itest += 1
            else:
                if train_feat_dset is None:
                    dataset_size = len(train_list)
                    C, F, D = feat.shape
                    # C, D = feat.shape
                    train_feat_dset = train_fd.create_dataset(feat_name, (dataset_size, C, F, D),
                                                      dtype=np.float32)
                    train_ids_dset = train_fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
                    itrain = 0

                train_feat_dset[itrain:(itrain + 1)] = feat
                train_ids_dset[itrain:(itrain + 1)] = int(vid)
                itrain += 1

def get_video_list(filename):
    samples = pd.read_csv(filename)
    videos = samples['video']
    videos = list(set(videos))
    print(len(videos))
    return sorted(videos)

def main():
    dataset = 'nextqa'
    data_dir = '../data/{}/'.format(dataset)
    dataset_dir = 'datasets/{}/'.format(dataset)
    # in_dir = osp.join(data_dir, 'region_n')
    out_dir = osp.join(data_dir, 'frame_feat')
    train_file = osp.join(dataset_dir, 'train.csv')
    val_file = osp.join(dataset_dir, 'val.csv')
    test_file = osp.join(dataset_dir, 'test.csv')
    train_list = get_video_list(train_file)
    val_list = get_video_list(val_file)
    test_list = get_video_list(test_file)

    # region features
    np2h5(osp.join(data_dir, 'frames_test'), out_dir, test_list, 'test')
    np2h5(osp.join(data_dir, 'frames_val'), out_dir, val_list, 'val')
    np2h5(osp.join(data_dir, 'frames_train'), out_dir, train_list, 'train')

    # appearance features
    h5filename = osp.join(out_dir, 'feat_appearance.h5')
    split_dataset_feat(h5filename, out_dir, train_list, val_list, test_list)


if __name__ == "__main__":
    main()