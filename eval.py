import sys
import os
import numpy as np
import copy
import open3d as o3d


FILE_ROOT = './TOSeg-Road-gt/'
SAVE_ROOT = './results/'

RED = np.array([237, 29,33])/255.0
GREEN = np.array([150, 255, 10])/255.0
BLUE = np.array([68, 114, 255])/255.0
BLACK = np.array([0, 0, 0])/255.0



def get_gt(dataset):
    file_stack = []
    FILE_ROOT_NEW = FILE_ROOT + dataset+'/'
    for file_name in sorted(os.listdir(FILE_ROOT_NEW)):
        file_path = os.path.join(FILE_ROOT_NEW, file_name)
        if file_path[-1] == 'd':
            file_stack.append(file_path)
    gt_stack = []
    for file in file_stack:
        with open(file, 'rb') as f:
            data = f.read()
            data_binary = data[data.find(b"DATA binary") + 12:]
            segs = np.frombuffer(data_binary, dtype=int)
            segs = segs.astype(np.float32)
            gt_stack.append(segs)
    gt_data = gt_stack
    return gt_data

def quality(method, dataset):
    # 1:green 2:red 3:pink 4:blue
    (TP, TN, FP, FN) = (0, 0, 0, 0)
    T = []
    gt_data = get_gt(dataset)
    if method == 'mrf':
        file_stack = []
        FILE_ROOT_NEW = SAVE_ROOT + dataset + '/'
        for file_name in sorted(os.listdir(FILE_ROOT_NEW)):
            file_path = os.path.join(FILE_ROOT_NEW, file_name)
            if file_path[-1] == 'd' :
                file_stack.append(file_path)
        for idx in range(len(gt_data)):
            seg_gt = np.asarray(gt_data[idx],dtype=int)
            seg_gt[seg_gt == 3] = 2
            file = file_stack[idx]
            if idx % 100 == 0:
                print('.',end='')
            pcd = o3d.io.read_point_cloud(file)
            colors = np.asarray(pcd.colors)[:, 0] * 255
            seg = np.zeros_like(colors)
            seg[(colors > 100) * (colors < 200)] = 1
            seg[colors > 200] = 2
            seg[colors < 100] = 4
            seg = np.asarray(seg,dtype=int)
            if seg.shape[0] != seg_gt.shape[0]:
                raise ValueError
            TP += sum((seg==seg_gt)*(seg!=1))
            TN += sum((seg==seg_gt)*(seg==1))
            FP += sum((seg!=seg_gt)*(seg!=1))
            FN += sum((seg!=seg_gt)*(seg==1))
            continue
    Acc = (TP + TN) / (TP + TN + FN + FP)
    R = TP / (TP + FN)
    P = TP / (TP + FP)
    F1 = 2 * P * R / (P + R)
    IOU = TP / (TP + FN + FP)
    return [Acc, P, R, F1, IOU]




if __name__ == '__main__':
    road_easy = quality('mrf', 'road_easy')
    print('road_easy_mrf:', road_easy)
    road_hard = quality('mrf', 'road_hard')
    print('road_hard_mrf:', road_hard)


