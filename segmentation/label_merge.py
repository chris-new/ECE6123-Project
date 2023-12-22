import numpy as np

def compute_bounding_box_overlapping_area(min_r_a, min_c_a, max_r_a, max_c_a, min_r_b, min_c_b, max_r_b, max_c_b):
    height = max(min(max_r_a, max_r_b) - max(min_r_a, min_r_b), 0)
    width = max(min(max_c_a, max_c_b) - max(min_c_a, min_c_b), 0)
    return height * width

def extend_left_right_coloring(position_mask, position_label, target_mask, left_label, right_label):
    '''
    "position" indicates the position of arms/legs to be colored.
    "target_mask" indicates the desired labels (with left and right information).

    First find out the positions that need to be recolored to show left or
    right (the positions of arms/legs). Then map these positions to the
    target_mask. Note that target_mask already have information about left and
    right. Lastly, extend such existing colored areas to fill the desired
    positions obtained in the first step.

    Note this method modifies target_mask in place.

    This method returns False when the condition is not desirable, i.e. when
    this method cannot work correctly. If it works successfully, it returns
    True.
    '''

    obtained_valid_results = True

    # offset for bounding boxes to ensure overlaps
    offset = 5

    # seperate two sets of arms/legs (find two clusters of arms/legs)
    h, w = position_mask.shape
    seperation_mask = np.zeros((h,w), dtype=np.bool8)
    inds = np.argwhere(position_mask == position_label)
    if len(inds) == 0:
        return False
    start_ind = inds[len(inds)//2]
    queue = [start_ind]
    min_r, min_c = start_ind
    max_r, max_c = start_ind
    seperation_mask[start_ind[0], start_ind[1]] = True
    while len(queue) > 0:
        row, col = queue.pop(0)
        for r in range(max(row-1,0), min(row+2,h)):
            for c in range(max(col-1,0), min(col+2,w)):
                if (not seperation_mask[r,c]) and (position_mask[r,c] == position_label):
                    seperation_mask[r,c] = True
                    queue.append((r,c))
                    min_r = min(min_r, r)
                    min_c = min(min_c, c)
                    max_r = max(max_r, r)
                    max_c = max(max_c, c)

    # find the left/right labels to the two sets
    inds = np.argwhere(target_mask == left_label)
    bb = min_r, min_c, max_r, max_c
    missing_left = False
    missing_right = False

    if len(inds) == 0: # no such labels for left side
        area_with_left = 0
        missing_left = True
    else:
        left_min_r = max(inds[:,0].min()-offset, 0)
        left_min_c = max(inds[:,1].min()-offset, 0)
        left_max_r = min(inds[:,0].max()+offset, h-1)
        left_max_c = min(inds[:,1].max()+offset, w-1)
        left_bb = left_min_r, left_min_c, left_max_r, left_max_c
        area_with_left = compute_bounding_box_overlapping_area(*bb, *left_bb)
    inds = np.argwhere(target_mask == right_label)

    if len(inds) == 0: # no such labels for right side
        area_with_right = 0
        missing_right = True
    else:
        right_min_r = max(inds[:,0].min()-offset, 0)
        right_min_c = max(inds[:,1].min()-offset, 0)
        right_max_r = min(inds[:,0].max()+offset, h-1)
        right_max_c = min(inds[:,1].max()+offset, w-1)
        right_bb = right_min_r, right_min_c, right_max_r, right_max_c
        area_with_right = compute_bounding_box_overlapping_area(*bb, *right_bb)

    if missing_left and missing_right:
        label1 = left_label
        label2 = left_label
        obtained_valid_results = False
    else:
        if area_with_left >= area_with_right:
            label1 = left_label
            label2 = right_label
        else:
            label1 = right_label
            label2 = left_label

    # assign the spacial labels
    set1 = seperation_mask & (position_mask == position_label)
    set2 = (~seperation_mask) & (position_mask == position_label)
    target_mask[set1] = label1
    target_mask[set2] = label2

    return obtained_valid_results

def get_left_right_labeled_results(pascal_results, atr_results, dataset_configs):
    '''
    Given two batches of segmentation masks (outputs from models based on Pascal
    Dataset and ATR Dataset), try to merget them together to get one batch of
    segmentation masks where left arms, right arms, left legs, and right legs
    are seperately labeled.

    In Pascal Dataset, upper arms, lower arms, upper legs, and lower arms can be
    labeled, but there is no left or right information.

    In ATR Dataset, most of the time, only left hands, right hands, left shoes,
    and right shoes can be labeled, but not arms or legs (when arms and legs are
    covered by clothing).
    '''
    pascal_lower_arms_label = dataset_configs['pascal']['labels'].index('Lower Arms')
    pascal_upper_arms_label = dataset_configs['pascal']['labels'].index('Upper Arms')
    pascal_lower_legs_label = dataset_configs['pascal']['labels'].index('Lower Legs')
    pascal_upper_legs_label = dataset_configs['pascal']['labels'].index('Upper Legs')
    atr_left_arms_label = dataset_configs['atr']['labels'].index('Left-arm')
    atr_right_arms_label = dataset_configs['atr']['labels'].index('Right-arm')
    atr_left_legs_label = dataset_configs['atr']['labels'].index('Left-shoe')
    atr_right_legs_label = dataset_configs['atr']['labels'].index('Right-shoe')
    # atr_left_arms_label = dataset_configs['lip']['labels'].index('Left-arm')
    # atr_right_arms_label = dataset_configs['lip']['labels'].index('Right-arm')
    # atr_left_legs_label = dataset_configs['lip']['labels'].index('Left-shoe')
    # atr_right_legs_label = dataset_configs['lip']['labels'].index('Right-shoe')

    target_left_arms_label = dataset_configs['target']['labels'].index('Left Arms')
    target_right_arms_label = dataset_configs['target']['labels'].index('Right Arms')
    target_left_legs_label = dataset_configs['target']['labels'].index('Left Legs')
    target_right_legs_label = dataset_configs['target']['labels'].index('Right Legs')

    combined_results = []
    for pascal_result, atr_result in zip(pascal_results, atr_results):
        combined_result = np.zeros(pascal_result.shape)

        # mapping from Pascal labels to customized labels
        # leave newly created labels as 0 (not exist in Pascal labels)
        for new_label, name in enumerate(dataset_configs['target']['labels']):
            if name in dataset_configs['pascal']['labels']:
                old_label = dataset_configs['pascal']['labels'].index(name)
                combined_result[pascal_result == old_label] = new_label

        # put initial left and right labels on the combined results
        # left and right information is obtained from ATR results
        combined_result[atr_result == atr_left_arms_label] = target_left_arms_label
        combined_result[atr_result == atr_right_arms_label] = target_right_arms_label
        combined_result[atr_result == atr_left_legs_label] = target_left_legs_label
        combined_result[atr_result == atr_right_legs_label] = target_right_legs_label

        combined_results.append(combined_result)

    # distinguish between left arms and right arms
    for i in range(len(pascal_results)):
        # color lower arms from hands
        extend_left_right_coloring(pascal_results[i], pascal_lower_arms_label, combined_results[i], target_left_arms_label, target_right_arms_label)
        # color lower legs from shoes
        extend_left_right_coloring(pascal_results[i], pascal_lower_legs_label, combined_results[i], target_left_legs_label, target_right_legs_label)
        # color upper arms from hands
        extend_left_right_coloring(pascal_results[i], pascal_upper_arms_label, combined_results[i], target_left_arms_label, target_right_arms_label)

    return combined_results