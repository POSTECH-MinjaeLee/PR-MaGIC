import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import pandas as pd 


def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--pred_path', type=str, default='persam')
    parser.add_argument('--dataset' , type=str , default='fss')
    parser.add_argument('--gt_path', type=str, default='./data/fss/Annotations')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--ref_idx', type=str, default='1')
    parser.add_argument('--nested', type= int, default=10)
    
    
    args = parser.parse_args()
    args.gt_path = f'./data/{args.dataset}/Annotations'
    return args


def main():

    args = get_arguments()
    print("Args:", args, "\n"), 
    table_path = os.path.join('./tables',args.pred_path) 
    #os.makedirs(table_path,exist_ok =True)
    class_names = sorted(os.listdir(args.gt_path))
    class_names = [class_name for class_name in class_names if ".DS" not in class_name]
    # should be replaced
    # class_names = ['can','candle','rc_car','robot_toy','teapot','fancy_boot']
    class_names.sort()

    # empty dataframe 
    df = pd.DataFrame(columns= [(t+1) for t in range(args.nested)], index= ['mIoU', 'mAcc', 'mDice']) 
    
    
   
    for t in range(1,args.nested+1):
        mIoU, mAcc, mDice = 0, 0, 0
        count = 0
        diff_classes = []  
        for class_name in class_names[args.start:args.end]:
            count += 1
            gt_path_class = os.path.join(args.gt_path, class_name)
            pred_path_class = os.path.join("./outputs/" + args.pred_path +'/' +str(t), class_name)

            gt_images = [str(img_path) for img_path in sorted(Path(gt_path_class).rglob("*.png"))]
            pred_images = [str(img_path) for img_path in sorted(Path(pred_path_class).rglob("*.png"))]

            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            total_meter = AverageMeter()

            for i, (gt_img, pred_img) in enumerate(zip(gt_images, pred_images)): 
                if args.ref_idx in gt_img:
                    continue
                gt_img = cv2.imread(gt_img)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY) > 0 
                gt_img = np.uint8(gt_img)

                pred_img = cv2.imread(pred_img)
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY) >0
                pred_img = np.uint8(pred_img)

                if pred_img.shape != gt_img.shape:
                    gt_img = cv2.resize(gt_img, pred_img.shape, interpolation=cv2.INTER_AREA)
                    diff_classes.append(class_name)
                
                intersection, union, target, total = intersectionAndUnion(pred_img, gt_img)
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target), total_meter.update(total)

            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            dice_class = intersection_meter.sum * 2 / (total_meter.sum + 1e-10)

            #print(class_name + ',', "IoU: %.2f," %(100 * iou_class), "Acc: %.2f" %(100 * accuracy_class), "Dice: %.2f\n" %(100 * dice_class))

            mIoU += iou_class
            mAcc += accuracy_class
            mDice += dice_class

        df[t] = {'mIoU' : 100 * mIoU / count, 'mAcc': 100 * mAcc / count, 'mDice' : 100 * mDice / count}
    print(df)
    df.to_csv(f'{table_path}.csv')
        # print("\nmIoU: %.2f" %(100 * mIoU / count))
        # print("mAcc: %.2f" %(100 * mAcc / count))
        # print("mDice: %.2f\n" %(100 * mDice / count))
    #print(diff_classes)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target):
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    
    area_total = output.sum()+target.sum()

    area_intersection = np.logical_and(output, target).sum()
    area_union = np.logical_or(output, target).sum()
    area_target = target.sum()
    
    return area_intersection, area_union, area_target, area_total


if __name__ == '__main__':
    main()
