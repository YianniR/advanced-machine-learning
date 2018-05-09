#THRESHOLDING

#use the previous threshold output from the function as the next min_threshold
def threshold_to_one_hot(continuous_piano_roll, min_threshold=0):
    sorted_pr_idx = sorted(range(len(continuous_piano_roll)), key=lambda k: continuous_piano_roll[k]) #largest last - gives the sorted indexes
    
    first  = continuous_piano_roll[sorted_pr_idx[-1]]
    second = continuous_piano_roll[sorted_pr_idx[-2]]
    third  = continuous_piano_roll[sorted_pr_idx[-3]]
    
    thresholded_piano_roll = np.zeros(128)
    if(first > min_threshold):
        thresholded_piano_roll[sorted_pr_idx[-1]] = 1
        if(second > 0.9*first):
            thresholded_piano_roll[sorted_pr_idx[-2]] = 1
            if(third> 0.9*first):
                thresholded_piano_roll[sorted_pr_idx[-3]] = 1
    
    return {'pianoroll':thresholded_piano_roll, 'nextthreshold':third}