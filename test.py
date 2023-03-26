from PIL import Image
import numpy as np
import os
import argparse


def confuse_matrix(score, lb, threshold=150):
    score_mask = (score>threshold)
    lb_pos_mask = (lb==255)
    lb_neg_mask = (lb==0) | (lb==50)

    tp = np.sum(lb_pos_mask * score_mask)
    fn = np.sum(lb_pos_mask * (~score_mask))
    tn = np.sum(lb_neg_mask * (~score_mask))
    fp = np.sum(lb_neg_mask * score_mask)
    
    return tp, fp, tn, fn

def eva_metrics(TP, FP, TN, FN):
    precision = TP/(TP+FP+1e-8)
    oa = (TP+TN)/(TP+FN+TN+FP+1e-8)
    recall = TP/(TP+FN+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    P = ((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2+1e-8)
    kappa = (oa-P)/(1-P+1e-8)
    
    return [precision, oa, recall, f1, kappa]

def test_score(datapath, category, f, sub_dir):
    fatherpath = os.path.join(datapath, category)
    subcategory = os.listdir(fatherpath)
    categorymetric = {}
    categorymetric['precision'] = []
    categorymetric['oa'] = []
    categorymetric['recall'] = []
    categorymetric['f1'] = []
    categorymetric['kappa'] = []
    for sub in subcategory:
        subTP = 0
        subFN = 0
        subTN = 0
        subFP = 0
        subpath = os.path.join(fatherpath, sub)
        maskpath = os.path.join(subpath, sub_dir)
        filelist = os.listdir(maskpath)
        for file_name in filelist:
            image_path = os.path.join(maskpath, file_name)
            if image_path[-3:] == 'jpg':
                pred = Image.open(image_path)
                if len(np.array(pred).shape) == 3:
                    pred = np.array(pred)[:,:,0]
                else:
                    pred = np.array(pred)
                num = image_path[-10:-4]
                gt_path = os.path.join(subpath, 'groundtruth', 'gt' + num + '.png')
                gt = Image.open(gt_path)
                gt = np.array(gt)
                if (gt==85).all() or (gt==170).all():
                    pass
                else:
                    tp, fp, tn, fn = confuse_matrix(pred, gt)
                    subTP += tp
                    subFN += fn
                    subTN += tn
                    subFP += fp
        precision, oa, recall, f1, kappa = eva_metrics(subTP, subFP, subTN, subFN)
        f.write(category+' '+sub+' '+'{} {} {} {} {} {} {} {} {}'.format(subTP,subFN,subTN,subFP,precision,oa,recall,f1,kappa))
        categorymetric['precision'].append(precision)
        categorymetric['oa'].append(oa)
        categorymetric['recall'].append(recall)
        categorymetric['f1'].append(f1)
        categorymetric['kappa'].append(kappa)
        f.write('\n')
        print(category, sub, 'precision:', precision, ',oa:', oa, ',recall:', recall, ',f1:', f1, ',kappa:', kappa)
    f.write(category+' all '+'- - - - {} {} {} {} {}'.format(np.mean(categorymetric['precision']),
                                                            np.mean(categorymetric['oa']),
                                                            np.mean(categorymetric['recall']),
                                                            np.mean(categorymetric['f1']),
                                                            np.mean(categorymetric['kappa'])))
    f.write('\n')
    print(category, 'all', 'precision:', np.mean(categorymetric['precision']), ',oa:', np.mean(categorymetric['oa']), ',recall:', np.mean(categorymetric['recall']), ',f1:', np.mean(categorymetric['f1']), ',kappa:', np.mean(categorymetric['kappa']))
    print('\n')
    return [np.mean(categorymetric['precision']), np.mean(categorymetric['oa']), np.mean(categorymetric['recall']), np.mean(categorymetric['f1']), np.mean(categorymetric['kappa'])]

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--sub_dir",
        type=str,
    )
    return parser


if __name__=='__main__':
    datapath = 'datasets/cdnet2014'
    args = get_parser().parse_args()
    categories = os.listdir(datapath)
    output_dir = 'log' + args.sub_dir[:-1] + '.txt'
    print(output_dir)
    with open(output_dir, 'w') as f:
        f.write('category subcategorty TP FN TN FP precision oa recall f1 kappa')
        f.write('\n')
        cls_list = []
        for cate in categories:
            category_res = test_score(datapath, cate, f, args.sub_dir[1:])
            cls_list.append(category_res)
        all_avg = np.mean(cls_list, axis=0)
        f.write('Total all '+'- - - - {} {} {} {} {}'.format(all_avg[0],all_avg[1],all_avg[2],all_avg[3],all_avg[4]))
    print('Total all', 'precision:', all_avg[0], ',oa:', all_avg[1], ',recall:', all_avg[2], ',f1:', all_avg[3], ',kappa:', all_avg[4])
