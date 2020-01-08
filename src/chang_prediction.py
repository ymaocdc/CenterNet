import os
import pandas as pd
from tqdm import tqdm
import numpy as np
def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        s.append(str(c))
    return ' '.join(s)

if __name__ == '__main__':
    root_dir = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/submission'
    csv_name = 'pku_everything_equalweight_5121024_flip_optim_fix_nms.csv'
    csv_file = os.path.join(root_dir, csv_name)
    train = pd.read_csv(csv_file)

    flip_csv = os.path.join('/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku', 'flipped.csv')
    flip_ids = pd.read_csv(flip_csv, header=None)

    flip_ids = list(flip_ids[0])

    image_ids = []

    prediction =[]
    count =0
    ss =[]
    for id in tqdm(range(len(train))):  # len(train)):
        gt = {'objects': []}

        img_name = train.loc[id]['ImageId']
        pred_string = train.loc[id]['PredictionString']

        image_ids.append(img_name)

        if type(pred_string) == np.float:
            prediction.append(pred_string)
        else:
            pred_string = pred_string[1:]

            items = pred_string.split(' ')
            tmp_pred = []
            yaws, pitches, rolls, xs, ys, zs, scores = [items[i::7] for i in range(7)]
            # overlay = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
            for yaw, pitch, roll, x, y, z, score in zip(yaws, pitches, rolls, xs, ys, zs, scores):
                # if float(z) <= 100:
                #     s = [yaw, pitch, roll, x, y, z, score]
                #     s = ' '.join(s)
                #     tmp_pred.append(s)
                # else:
                #     count += 1
                # if float(y) <0:
                #     print(y)

                if img_name+'.jpg' in flip_ids:
                    roll = '-3.1'
                else:
                    roll= '-3.10'
                if float(score) >= 0.6:
                    s = [yaw, pitch, roll, x, y, z, score]
                    ss.append(float(x))
                    tmp_pred.append(' '.join(s))
                else:
                    count = count+1
            tmp_pred = ' '.join(tmp_pred)

            prediction.append(tmp_pred)
    print(count)
    ss = np.array(ss)
    data = {
        'ImageId': image_ids,
        'PredictionString': prediction,
    }
    val_df = pd.DataFrame(data=data)

    save_to = os.path.join(root_dir, '{}_{}'.format('score06', csv_name))
    val_df.to_csv(save_to, index=False)
    print(save_to)