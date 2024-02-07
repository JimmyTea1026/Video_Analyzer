import os
import pickle
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib import pyplot as plt

def brighness_plot():
    daylight = pickle.load(open('/app/result/daylight.pkl', 'rb'))
    night = pickle.load(open('/app/result/night.pkl', 'rb'))
    daylight_list = []
    night_list = []
    for l in daylight:
        filtered = [item for item in l['brightness'] if item is not None]
        daylight_list.append(filtered)
    for l in night:
        filtered = [item for item in l['brightness'] if item is not None]
        night_list.append(filtered)
        
    m1 = []
    m2 = []
    std1 = []
    std2 = []
    for i in daylight_list:
        m1.append(np.mean(i))
        std1.append(np.std(i))
    for i in night_list:
        m2.append(np.mean(i))
        std2.append(np.std(i))
    print("Daylight", np.mean(m1), np.std(m1))
    print("Night", np.mean(m2), np.std(m2))
    plt.errorbar(range(len(m1)), m1, yerr=std1, fmt='o', capsize=5, label='Mean ± Std', ecolor='red')
    plt.errorbar(range(len(m1), len(m1)+len(m2)), m2, yerr=std2, fmt='o', capsize=5, label='Mean ± Std', ecolor='black')
    #-----plot probability density function-----#
    # plt.hist(m1, bins=20, alpha=0.5, label='Daylight', density=True, edgecolor='black')
    # plt.hist(m2, bins=20, alpha=0.5, label='Night', density=True, edgecolor='black')
    # data1 = np.random.normal(np.mean(m1), np.std(m2), 1000)
    # data2 = np.random.normal(np.mean(m2), np.std(m2), 1000)
    # kde1 = gaussian_kde(data1, bw_method=0.5)
    # x1 = np.linspace(min(data1), max(data1), 1000)
    # y1 = kde1(x1)
    # kde2 = gaussian_kde(data2, bw_method=0.5)
    # x2 = np.linspace(min(data2), max(data2), 1000)
    # y2 = kde2(x2)
    # plt.plot(x1, y1, label='Data 1')
    # plt.plot(x2, y2, label='Data 2')
    plt.savefig('/app/result/bright.png')
    
def rotate_plot():
    for t in ['mask', 'no_mask']:
        result_dict_list = pickle.load(open(f'/app/result/{t}.pkl', 'rb'))
        save_path = "/app/result/rotate"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        mean_yaw_list = []
        std_yaw_list = []
        # mean_pitch_list = []
        # std_pitch_list = []
        # std_roll_list = []
        
        for i, result_dict in enumerate(result_dict_list):
            rotate_list = result_dict['rotate']
            yaw = [abs(point[0]) for point in rotate_list]
            pitch = [abs(point[1]) for point in rotate_list]
            roll = [abs(point[2]) for point in rotate_list]
            mean_yaw = np.mean(yaw)
            std_yaw = np.std(yaw)
            mean_yaw_list.append(mean_yaw)
            std_yaw_list.append(std_yaw)
        if t == 'mask': color = 'red'
        else: color = 'blue'
        
        plt.scatter(std_yaw_list, mean_yaw_list, label=t, color=color)
    plt.savefig(f'{save_path}/rotate.png')

def mask_accu_test():
    '''
    moving box : 十幀內有6幀以上的mask則判斷為mask
    '''
    box = 10
    limit = 6
    mask_dict_list = pickle.load(open('/app/result/mask.pkl', 'rb'))
    no_mask_dict_list = pickle.load(open('/app/result/no_mask.pkl', 'rb'))
    predict_mask = 0
    for mask_dict in mask_dict_list:
        mask_list = mask_dict['mask']
        for i in range(0, len(mask_list)-box):
            mv = mask_list[i:i+box]
            count = mv.count(True)
            if count >= limit:
                predict_mask += 1
                break
    tp = round(predict_mask/len(mask_dict_list),2)
    print(f'true positive : {tp}')
    predict_mask = 0
    for no_mask_dict in no_mask_dict_list:
        no_mask_list = no_mask_dict['mask']
        for i in range(0, len(no_mask_list)-box):
            mv = no_mask_list[i:i+box]
            count = mv.count(True)
            if count >= limit:
                predict_mask += 1
                break
    fp = round(predict_mask/len(no_mask_dict_list),2)
    print(f'false positive : {fp}')
                
if __name__ == '__main__':
    # brighness_plot()
    rotate_plot()
    # mask_accu_test()