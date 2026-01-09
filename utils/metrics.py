"""
metrics.py

This module defines evaluation metrics to monitor the model's performance
during training and testing.

Author: David Diaz-Guerra, Audio Research Group, Tampere University
Date: February 2025
"""

import pandas as pd
import numpy as np
import os
import warnings

import torch

# def conversion360_90():


def save_pred(pred_data,th_prob,save_folder,file_name):
    for i in range(pred_data.size(0)):
        pred_data_i = pred_data[i]
        results = []
        for j in range(pred_data_i.size(0)):
            pred_angle_label = torch.atan2(pred_data_i[j,:,1], pred_data_i[j, :, 0]) / torch.pi * 180
            class_idx = []
            for k in range(pred_data_i.size(1)):
                pred_class_prob = torch.norm(torch.tensor([pred_data_i[j,k,0],pred_data_i[j,k,1]]))
                if pred_class_prob >= th_prob:
                    class_idx.append(k)
            pred_angle_label2 = pred_angle_label[class_idx]
            if pred_angle_label2.numel() == 1:
                results.append([j,class_idx[0],pred_angle_label2.item()])
            elif pred_angle_label2.numel() > 1:
                for l in range(pred_angle_label2.numel()):
                    results.append([j, class_idx[l], pred_angle_label2[l].item()])
                    c = 1
        df = pd.DataFrame(results, columns=["frame","class", "sfx_azimuth"])
        df.to_csv(save_folder + file_name[i][:-3] + "csv", index=False, encoding="utf-8")

def tensor_label2dict_label(label_accdoa):
    # label_accdoa: [B,T,class_number,accdoa_x,accdoa_y]
    label_data_dict_list = []
    for i in range(label_accdoa.size(0)):
        label_data_dict = {}
        for j in range(label_accdoa.size(1)):
            for k in range(label_accdoa.size(2)):
                norm_label = torch.norm(torch.tensor([label_accdoa[i,j,k,0],label_accdoa[i,j,k,1]]))
                if norm_label < 0.5:
                    continue
                else:
                    angle_label = torch.atan2(label_accdoa[i,j,k,1],label_accdoa[i,j,k,0])/torch.pi*180
                    frame_idx = j
                    data_row = [k, angle_label]
                    if frame_idx not in label_data_dict:
                        label_data_dict[frame_idx] = []
                    label_data_dict[frame_idx].append(data_row)
        label_data_dict_list.append(label_data_dict)

    return label_data_dict_list


def load_labels(label_file):
    label_data = {}
    with open(label_file, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
        for line in lines:
            values = line.strip().split(',')
            frame_idx = int(values[0])
            data_row = [int(values[1]), float(values[2])]
            if frame_idx not in label_data:
                label_data[frame_idx] = []
            label_data[frame_idx].append(data_row)
    return label_data

def organize_labels(input_dict, max_frames, max_tracks=10):
    """
    :param input_dict: Dictionary containing frame-wise sound event time and location information
            _pred_dict[frame-index] = [[class-index, source-index, azimuth, distance, onscreen] x events in frame]
    :param max_frames: Total number of frames in the recording
    :param max_tracks: Total number of tracks in the output dict
    :return: Dictionary containing class-wise sound event location information in each frame
            dictionary_name[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
    """
    tracks = set(range(max_tracks))
    output_dict = {x: {} for x in range(max_frames)}
    for frame_idx in range(0, max_frames):
        if frame_idx not in input_dict:
            continue
        for [class_idx, az] in input_dict[frame_idx]:
            if class_idx not in output_dict[frame_idx]:
                output_dict[frame_idx][class_idx] = {}
            else:                       # If not, use the first one available
                try:
                    track_idx = list(set(tracks) - output_dict[frame_idx][class_idx].keys())[0]
                except IndexError:
                    warnings.warn("The number of sources of is higher than the number of tracks. "
                                  "Some events will be missed.")
                    track_idx = 0  # Overwrite one event
            if az > 270:
                az = az - 360
            elif az > 90:
                az = 180 - az
            output_dict[frame_idx][class_idx][0] = [az]

    return output_dict



class SELDMetrics(object):

    def __init__(self, doa_threshold=20,nb_classes=13, average='macro'):
        """
        This class implements both the class-sensitive localization and location-sensitive detection metrics.
        只保留角度误差的计算，并且一个类只有一个轨

        :param doa_threshold: DOA error threshold for location sensitive detection.
        :param nb_classes: Number of sound classes.
        :param average: Whether 'macro' or 'micro' aggregate the results
        """
        self._nb_classes = nb_classes

        # Variables for Location-sensitive detection performance
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)

        self._Nref = np.zeros(self._nb_classes) # 每个类的轨道数，当前SELD模型仅支持单轨

        self._ang_T = doa_threshold

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        self._total_AngE = np.zeros(self._nb_classes)
        self._total_DistE = np.zeros(self._nb_classes)
        self._total_RelDistE = np.zeros(self._nb_classes)
        self._total_OnscreenCorrect = np.zeros(self._nb_classes)

        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

        assert average in ['macro', 'micro'], "Only 'micro' and 'macro' average are supported"
        self._average = average

    def compute_seld_scores(self):
        """
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores:
            F score, angular error, distance error, relative distance error, onscreen accuracy, and classwise results
        """
        eps = np.finfo(float).eps
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (
                        eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            AngE = self._total_AngE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.nan

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            AngE = self._total_AngE / (self._DE_TP + eps)
            AngE[self._DE_TP == 0] = np.nan

            classwise_results = np.array([F, AngE])
            F, AngE = F.mean(), np.nanmean(AngE)
        else:
            raise NotImplementedError('Only micro and macro averaging are supported.')

        return F, AngE, classwise_results

    def update_seld_scores(self, pred, gt):
        """
        Computes the SELD scores given a prediction and ground truth labels.

        :param pred: dictionary containing the predictions for every frame
            pred[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
        :param gt: dictionary containing the ground truth for every frame
            gt[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
        """
        eps = np.finfo(float).eps

        for frame_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of reference tracks for each class，当前每个类只有1个track
                try:
                    # 可能出错的代码
                    nb_gt_doas = len(gt[frame_cnt][class_cnt]) if class_cnt in gt[frame_cnt] else None
                except Exception as e:
                    c = 1
                nb_pred_doas = len(pred[frame_cnt][class_cnt]) if class_cnt in pred[frame_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[frame_cnt] and class_cnt in pred[frame_cnt]:
                    # True positives

                    # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                    # DOA-tracks using hungarian algorithm on the azimuth estimation and then compute the average
                    # spatial distance between the associated reference-predicted tracks.

                    gt_az = np.array(list(gt[frame_cnt][class_cnt].values())) # 当前仅预测事件类别和azimuth
                    pred_az = np.array(list(pred[frame_cnt][class_cnt].values()))

                    # Reference and predicted track matching
                    doa_err_list = np.abs(gt_az - pred_az)

                    # https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#evaluation
                    Pc = len(pred_az)
                    Rc = len(gt_az)
                    FNc = max(0, Rc - Pc)
                    FPcinf = max(0, Pc - Rc)
                    Kc = min(Pc, Rc)
                    TPc = Kc
                    Lc = np.sum(np.any((doa_err_list > self._ang_T),axis=0))
                    FPct = Lc
                    FPc = FPcinf + FPct
                    TPct = Kc - FPct
                    assert Pc == TPct + FPc
                    assert Rc == TPct + FPct + FNc

                    self._total_AngE[class_cnt] += doa_err_list.sum()

                    self._TP[class_cnt] += TPct
                    self._DE_TP[class_cnt] += TPc

                    self._FP[class_cnt] += FPcinf
                    self._DE_FP[class_cnt] += FPcinf
                    self._FP_spatial[class_cnt] += FPct
                    loc_FP += FPc

                    self._FN[class_cnt] += FNc
                    self._DE_FN[class_cnt] += FNc
                    loc_FN += FNc

                elif class_cnt in gt[frame_cnt] and class_cnt not in pred[frame_cnt]:
                    # False negative
                    loc_FN += nb_gt_doas
                    self._FN[class_cnt] += nb_gt_doas
                    self._DE_FN[class_cnt] += nb_gt_doas
                elif class_cnt not in gt[frame_cnt] and class_cnt in pred[frame_cnt]:
                    # False positive
                    loc_FP += nb_pred_doas
                    self._FP[class_cnt] += nb_pred_doas
                    self._DE_FP[class_cnt] += nb_pred_doas
                else:
                    # True negative
                    pass


class ComputeSELDResults(object):
    def __init__(self, params, ref_files_folder=None):
        """
        This class takes care of computing the SELD scores from the reference and predicted csv files.

        :param params: Dictionary containing the parameters of the SELD evaluation.
        :param ref_files_folder: Folder containing the split folders with the reference csv files.
        """
        self._desc_dir = ref_files_folder

        self._doa_thresh = params['lad_doa_thresh']

        # collect reference files
        self._ref_labels = {}
        for ref_file in os.listdir(os.path.join(self._desc_dir,)):
            # Load reference description file
            gt_dict = load_labels(os.path.join(self._desc_dir, ref_file))
            nb_ref_frames = max(list(gt_dict.keys())) if len(gt_dict) > 0 else 0
            self._ref_labels[ref_file] = [organize_labels(gt_dict, nb_ref_frames),
                                              nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['average']
        self._nb_classes = params['num_classes']

    def get_SELD_Results2(self, batch_pred, batch_labels):
        """
        Compute the SELD scores for the predicted csv files in a given folder.

        :param pred_files_path: Folder containing the predicted csv files.
        """
        # collect predicted files info
        eval = SELDMetrics(doa_threshold=self._doa_thresh,
                           nb_classes=self._nb_classes, average=self._average)
        pred_labels_dict = {}
        for i in range(len(batch_pred)):
            # Calculated scores
            pred_labels = batch_pred[i]
            ref_labels = batch_labels[i]
            eval.update_seld_scores(pred_labels, ref_labels)
        # Overall SED and DOA scores
        F, AngE, classwise_results = eval.compute_seld_scores()
        return (F, AngE, classwise_results)

    def get_SELD_Results(self, pred_files_path,is_jackknife=False):
        """
        Compute the SELD scores for the predicted csv files in a given folder.

        :param pred_files_path: Folder containing the predicted csv files.
        :param is_jackknife: Whether to compute the Jackknife confidence intervals.
        """
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        eval = SELDMetrics(doa_threshold=self._doa_thresh,
                           nb_classes=self._nb_classes, average=self._average)
        pred_labels_dict = {}
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = load_labels(os.path.join(pred_files_path, pred_file))
            nb_pred_frames = max(list(pred_dict.keys())) if len(pred_dict) > 0 else 0
            nb_ref_frames = self._ref_labels[pred_file][1]
            pred_labels = organize_labels(pred_dict, max(nb_pred_frames, nb_ref_frames))

            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])
            if is_jackknife:
                pred_labels_dict[pred_file] = pred_labels
        # Overall SED and DOA scores
        F, AngE, classwise_results = eval.compute_seld_scores()

        if is_jackknife:
        #     global_values = [F, AngE]
        #     if len(classwise_results):
        #         global_values.extend(classwise_results.reshape(-1).tolist())
        #     partial_estimates = []
        #     # Calculate partial estimates by leave-one-out method
        #     for leave_file in pred_files:
        #         leave_one_out_list = pred_files[:]
        #         leave_one_out_list.remove(leave_file)
        #         eval = SELDMetrics(doa_threshold=self._doa_thresh,
        #                            nb_classes=self._nb_classes, average=self._average)
        #         for pred_cnt, pred_file in enumerate(leave_one_out_list):
        #             # Calculated scores
        #             eval.update_seld_scores(pred_labels_dict[pred_file], self._ref_labels[pred_file][0])
        #         F, AngE, classwise_results = eval.compute_seld_scores()
        #         leave_one_out_est = [F, AngE, classwise_results]
        #         if len(classwise_results):
        #             leave_one_out_est.extend(classwise_results.reshape(-1).tolist())
        #
        #         # Overall SED and DOA scores
        #         partial_estimates.append(leave_one_out_est)
        #     partial_estimates = np.array(partial_estimates)
        #
        #     estimate, bias = [-1] * len(global_values), [-1] * len(global_values)
        #     std_err, conf_interval = [-1] * len(global_values), [-1] * len(global_values)
        #     for i in range(len(global_values)):
        #         estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
        #             global_value=global_values[i],
        #             partial_estimates=partial_estimates[:, i],
        #             significance_level=0.05
        #         )
        #     # return ([F, conf_interval[0]], [AngE, conf_interval[1]], [DistE, conf_interval[2]],
        #     #         [RelDistE, conf_interval[3]], [OnscreenAq, conf_interval[4]],
        #     #         [classwise_results, np.array(conf_interval)[5:].reshape(5, 13, 2) if len(classwise_results) else []])
            return (F, AngE, classwise_results)

        else:
            return (F, AngE, classwise_results)


if __name__ == '__main__':
    # use this to test if the metrics class works as expected. All the classes will be called from the main.py for
    # actual use
    pred_output_files = 'outputs/SELD_fake_estimates/dev-test'  # Path of the DCASE output format files
    from parameters import params
    # Compute just the DCASE final results
    use_jackknife = False
    eval_dist = params['evaluate_distance'] if 'evaluate_distance' in params else False
    score_obj = ComputeSELDResults(params, ref_files_folder='../DCASE2025_SELD_dataset/metadata_simple_header_int_dev')
    F, AngE, classwise_test_scr = score_obj.get_SELD_Results(pred_output_files,
                                                                                          is_jackknife=use_jackknife)
    print('SED F-score: {:0.1f}% {}'.format(100 * F[0] if use_jackknife else 100 * F,
                                            '[{:0.2f}, {:0.2f}]'.format(100 * F[1][0], 100 * F[1][1])
                                            if use_jackknife else ''))
    print('DOA error: {:0.1f} {}'.format(AngE[0] if use_jackknife else AngE,
                                         '[{:0.2f}, {:0.2f}]'.format(AngE[1][0], AngE[1][1])
                                         if use_jackknife else ''))

    if params['average'] == 'macro':
        print('Classwise results on unseen test data')
        print('Class\tF\tAngE\tDistE\tRelDistE\tOnscreenAq')
        for cls_cnt in range(params['nb_classes']):
            print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                cls_cnt,
                classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                                            classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                            classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                            classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                            classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                            classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))


