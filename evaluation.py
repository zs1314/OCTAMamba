import numpy as np
import math
import cv2
# from sklearn.metrics import roc_auc_score

def calc_result(np_pred:np.ndarray, np_label:np.ndarray, thresh_value=None):
    temp = cv2.normalize(np_pred, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    if thresh_value==None:
        _,np_pred = cv2.threshold(temp, 0.0, 1.0, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _,np_pred = cv2.threshold(temp, thresh_value, 1.0, cv2.THRESH_BINARY)


    np_pred = np_pred.flatten()
    np_label = np_label.flatten()



    uni = np.unique(np_label)
    # print(uni)
    assert (len(uni)==2) and (1 in uni) and (0 in uni)


    FP = np.sum(np.logical_and(np_pred == 1, np_label == 0)).astype(float)
    FN = np.sum(np.logical_and(np_pred == 0, np_label == 1)).astype(float)
    TP = np.sum(np.logical_and(np_pred == 1, np_label == 1)).astype(float)
    TN = np.sum(np.logical_and(np_pred == 0, np_label == 0)).astype(float)

    result = {}
    smooth = 1e-12
    # result['auc'] = roc_auc_score(np_label, np_pred)
    result['acc'] = (TP + TN) / (FP + FN + TP + TN)
    result['fdr'] = (FP + smooth)  / (FP + TP + smooth)
    sen = (TP + smooth) / (FN + TP + smooth)
    spe = (TN + smooth) / (FP + TN + smooth)
    result['sen'] = sen
    result['spe'] = spe
    result['gmean'] = math.sqrt(sen * spe)
    
    # result['kappa'] = calc_kappa(pred, gt)
    
    result['iou'] = (TP + smooth) / (FP + FN + TP + smooth)
    result['dice'] = (2.0 * TP + smooth) / (FP + FN + 2.0 * TP + smooth)
    
    return result

def avg_result(ls_result):
    total_result = {}
    for r in ls_result:
        for key in r:
            if key not in total_result:
                total_result[key] = []
            total_result[key].append(r[key])
    for key in total_result:
        values = np.array(total_result[key])
        # print(f"{key} - mean: {values.mean()} \t std: {values.std()}")
        total_result[key] = float(values.mean())
    return total_result