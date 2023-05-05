def compute_eer(target_scores, nontarget_scores): 
    if isinstance(target_scores , list) is False:
        target_scores = list(target_scores)
    if isinstance(nontarget_scores , list) is False:
        nontarget_scores = list(nontarget_scores)

    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)
    target_size = len(target_scores);
    nontarget_size = len(nontarget_scores)
    for i in range(target_size-1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break

    th = target_scores[target_position];
    eer = target_position * 1.0 / target_size;
    return eer, th

def calc_frr_at_far_percent(target_scores, nontarget_scores, far_th):
    far_percent = far_th * 100

    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    nontarget_size = len(nontarget_scores)
    nontarget_n = nontarget_size * far_percent / 100.0
    nontarget_position = int(nontarget_size - 1 - nontarget_n)

    threshold = nontarget_scores[nontarget_position]
    cnt = 0;
    for i in range(len(target_scores)):
        if target_scores[i] < threshold:
            cnt += 1
        else:
            break

    frr = cnt * 1.0 / len(target_scores)
    return frr

def cal_frr_at_fix_far(score_true, score_false, far_th): 
    fpr, tpr, thresholds = metrics.roc_curve([1]*len(score_true)+[0]*len(score_false), score_true+score_false, pos_label=1) 
    frr = 1 - tpr

    res = (-1, -1)
    min_diff = 1e8
    for i in range(len(fpr)):
        diff = abs(far_th - fpr[i])
        if diff < min_diff:
            min_diff = diff
            res = (fpr[i], frr[i])
    return res
