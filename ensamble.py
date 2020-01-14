import os
import pandas as pd
import numpy as np
def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        s.append(str(c))
    return ' '.join(s)

root_dir = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku'
dir = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/submission'
files = [ 'dla_34_0scaleaugtest_results.csv','dla_34_q10_shift01_comb_nms_09.csv']

test = pd.read_csv(os.path.join(root_dir, 'sample_submission.csv'))

detections = []
for file in files:
    res = pd.read_csv(os.path.join(dir, file))
    detections.append(res)

predictions = []
for idx in range(len(detections[0])):
    prediction_per_image = []
    for res in detections:
        img_name = res.loc[idx]['ImageId']
        pred_string = res.loc[idx]['PredictionString']
        try:
            prediction_per_image += pred_string.split(' ')
        except:
            print(pred_string)

    yaw, pitch, roll, x, y, z, s = [prediction_per_image[i::7] for i in range(7)]
    prediction = np.array([yaw, pitch, roll, x, y, z, s], dtype=np.float).T
    check = np.zeros(len(yaw))

    merged_prediction = []
    for c_idx in range(len(yaw)):
        if check[c_idx] == 0:
            cyaw, cp, cr, cx, cy, cz, cs = prediction[c_idx]
            check[c_idx] = 1
            match = [c_idx]
            for comp_idx in range(c_idx+1, len(yaw)):
                if check[comp_idx] == 0:
                    comyaw, comp, comr, comx, comy, comz, coms = prediction[comp_idx]
                    dis = np.sqrt(((np.array([cx, cy, cz]) - np.array([comx, comy, comz]))**2).sum())
                    if dis < cz/10:
                        check[comp_idx] = 1
                        match.append(comp_idx)
            merge = prediction[match].mean(axis=0)
            merge[-1] = prediction[match][:,-1].max()
            merged_prediction.append(merge)

    pstr = []
    for p in merged_prediction:
        pstr.append(coords2str(p))
    pstr = ' '.join(pstr)
    test['PredictionString'][idx] = pstr

test.to_csv(os.path.join(dir, 'ensemble.csv'), index=False)