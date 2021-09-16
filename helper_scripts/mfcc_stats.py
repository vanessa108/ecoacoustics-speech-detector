## STEP 2
import os
import librosa
from librosa.core import audio
import numpy as np
import pandas as pd
from scipy import stats
from pandas.core.arrays.categorical import contains

main_dir = 'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-train'
# 'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2'
#'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-train'

input_folder = main_dir+'/cv-train-mix-chunks' # folder with chunks
proc_file = main_dir+'/cv-valid-processed-mfcc.csv'
#'processed_files.csv'
#main_dir+'/cv-valid-processed-mfcc.csv' # folder with list of mfccs that have been calculated
main_file =  main_dir + '/cv_train_main_file.csv'
#'main_file.csv'
# main_dir + '/cv_train_main_file.csv' # main file with labels and mfccs

processed = pd.read_csv(proc_file, header=None) 
processed_files = processed.squeeze().tolist()
df = pd.read_csv(main_file)


def audio2MFCCFeatures(audio):
    signal, sr = librosa.load(audio)
    mfccs = librosa.feature.mfcc(y = signal, sr =sr, n_mfcc=13)
    mfccs_d = librosa.feature.delta(mfccs)
    mfccs_dd = librosa.feature.delta(mfccs, order=2)
    #features = np.concatenate((mfccs, mfccs_d, mfccs_dd))
    mfcc_mean, mfcc_median, mfcc_max, mfcc_min, mfcc_std, mfcc_var, mfcc_skew, mfcc_kurtosis = [np.zeros(13)]*8
    dmfcc_mean, dmfcc_median, dmfcc_max, dmfcc_min, dmfcc_std, dmfcc_var, dmfcc_skew, dmfcc_kurtosis = [np.zeros(13)]*8
    ddmfcc_mean, ddmfcc_median, ddmfcc_max, ddmfcc_min, ddmfcc_std, ddmfcc_var, ddmfcc_skew, ddmfcc_kurtosis = [np.zeros(13)]*8
    for i in range(13):
        thisRow = mfccs[i, :]
        dThisRow = mfccs_d[i, :]
        ddThisRow = mfccs_dd[i, :]
        mfcc_mean[i] = np.average(thisRow); mfcc_median[i] = np.median(thisRow)
        mfcc_max[i] = np.max(thisRow); mfcc_min[i] = np.min(thisRow)
        mfcc_std[i] = np.std(thisRow); mfcc_var[i] = np.var(thisRow)
        mfcc_skew[i] = stats.skew(thisRow); mfcc_kurtosis[i] = stats.kurtosis(thisRow)
        dmfcc_mean[i] = np.average(dThisRow); dmfcc_median[i] = np.median(dThisRow)
        dmfcc_max[i] = np.max(dThisRow); dmfcc_min[i] = np.min(dThisRow)
        dmfcc_std[i] = np.std(dThisRow); dmfcc_var[i] = np.var(dThisRow)
        dmfcc_skew[i] = stats.skew(dThisRow); dmfcc_kurtosis[i] = stats.kurtosis(dThisRow)
        ddmfcc_mean[i] = np.average(ddThisRow); ddmfcc_median[i] = np.median(ddThisRow)
        ddmfcc_max[i] = np.max(ddThisRow); ddmfcc_min[i] = np.min(ddThisRow)
        ddmfcc_std[i] = np.std(ddThisRow); ddmfcc_var[i] = np.var(ddThisRow)
        ddmfcc_skew[i] = stats.skew(ddThisRow); ddmfcc_kurtosis[i] = stats.kurtosis(ddThisRow)

    return np.hstack([mfcc_mean, mfcc_median, mfcc_max, mfcc_min, mfcc_std, mfcc_var, mfcc_skew, mfcc_kurtosis, \
        dmfcc_mean, dmfcc_median, dmfcc_max, dmfcc_min, dmfcc_std, dmfcc_var, dmfcc_skew, dmfcc_kurtosis, \
            ddmfcc_mean, ddmfcc_median, ddmfcc_max, ddmfcc_min, ddmfcc_std, ddmfcc_var, ddmfcc_skew, ddmfcc_kurtosis])



for (dirpath, dirnames, filenames) in os.walk(input_folder):
    for filename in filenames:
        if filename.endswith('.wav') and str(filename) not in processed_files:
            filepath = input_folder + '/' + filename
            # try:
            features = audio2MFCCFeatures(filepath)
            df.loc[df['filename'].str.match(filename), ['mean0', 'mean1', 'mean2', 'mean3', 'mean4', 'mean5', 'mean6',\
                'mean7', 'mean8', 'mean9', 'mean10', 'mean11', 'mean12', 'median0', 'median1', 'median2', 'median3', 'median4', 'median5', 'median6',\
                'median7', 'median8', 'median9', 'median10', 'median11', 'median12','max0', 'max1', 'max2', 'max3', 'max4', 'max5', 'max6',\
                'max7', 'max8', 'max9', 'max10', 'max11', 'max12','min0', 'min1', 'min2', 'min3', 'min4', 'min5', 'min6',\
                'min7', 'min8', 'min9', 'min10', 'min11', 'min12','std0', 'std1', 'std2', 'std3', 'std4', 'std5', 'std6',\
                'std7', 'std8', 'std9', 'std10', 'std11', 'std12','var0', 'var1', 'var2', 'var3', 'var4', 'var5', 'var6',\
                'var7', 'var8', 'var9', 'var10', 'var11', 'var12','skew0', 'skew1', 'skew2', 'skew3', 'skew4', 'skew5', 'skew6',\
                'skew7', 'skew8', 'skew9', 'skew10', 'skew11', 'skew12','kurtosis0', 'kurtosis1', 'kurtosis2', 'kurtosis3', 'kurtosis4', 'kurtosis5', 'kurtosis6',\
                'kurtosis7', 'kurtosis8', 'kurtosis9', 'kurtosis10', 'kurtosis11', 'kurtosis12',\
                'dmean0', 'dmean1', 'dmean2', 'dmean3', 'dmean4', 'dmean5', 'dmean6',\
                'dmean7', 'dmean8', 'dmean9', 'dmean10', 'dmean11', 'dmean12', 'dmedian0', 'dmedian1', 'dmedian2', 'dmedian3', 'dmedian4', 'dmedian5', 'dmedian6',\
                'dmedian7', 'dmedian8', 'dmedian9', 'dmedian10', 'dmedian11', 'dmedian12','dmax0', 'dmax1', 'dmax2', 'dmax3', 'dmax4', 'dmax5', 'dmax6',\
                'dmax7', 'dmax8', 'dmax9', 'dmax10', 'dmax11', 'dmax12','dmin0', 'dmin1', 'dmin2', 'dmin3', 'dmin4', 'dmin5', 'dmin6',\
                'dmin7', 'dmin8', 'dmin9', 'dmin10', 'dmin11', 'dmin12','dstd0', 'dstd1', 'dstd2', 'dstd3', 'dstd4', 'dstd5', 'dstd6',\
                'dstd7', 'dstd8', 'dstd9', 'dstd10', 'dstd11', 'dstd12','dvar0', 'dvar1', 'dvar2', 'dvar3', 'dvar4', 'dvar5', 'dvar6',\
                'dvar7', 'dvar8', 'dvar9', 'dvar10', 'dvar11', 'dvar12','dskew0', 'dskew1', 'dskew2', 'dskew3', 'dskew4', 'dskew5', 'dskew6',\
                'dskew7', 'dskew8', 'dskew9', 'dskew10', 'dskew11', 'dskew12','dkurtosis0', 'dkurtosis1', 'dkurtosis2', 'dkurtosis3', 'dkurtosis4', 'dkurtosis5', 'dkurtosis6',\
                'dkurtosis7', 'dkurtosis8', 'dkurtosis9', 'dkurtosis10', 'dkurtosis11', 'dkurtosis12',\
                    'ddmean0', 'ddmean1', 'ddmean2', 'ddmean3', 'ddmean4', 'ddmean5', 'ddmean6',\
                'ddmean7', 'ddmean8', 'ddmean9', 'ddmean10', 'ddmean11', 'ddmean12', 'ddmedian0', 'ddmedian1', 'ddmedian2', 'ddmedian3', 'ddmedian4', 'ddmedian5', 'ddmedian6',\
                'ddmedian7', 'ddmedian8', 'ddmedian9', 'ddmedian10', 'ddmedian11', 'ddmedian12','ddmax0', 'ddmax1', 'ddmax2', 'ddmax3', 'ddmax4', 'ddmax5', 'ddmax6',\
                'ddmax7', 'ddmax8', 'ddmax9', 'ddmax10', 'ddmax11', 'ddmax12','ddmin0', 'ddmin1', 'ddmin2', 'ddmin3', 'ddmin4', 'ddmin5', 'ddmin6',\
                'ddmin7', 'ddmin8', 'ddmin9', 'ddmin10', 'ddmin11', 'ddmin12','ddstd0', 'ddstd1', 'ddstd2', 'ddstd3', 'ddstd4', 'ddstd5', 'ddstd6',\
                'ddstd7', 'ddstd8', 'ddstd9', 'ddstd10', 'ddstd11', 'ddstd12','ddvar0', 'ddvar1', 'ddvar2', 'ddvar3', 'ddvar4', 'ddvar5', 'ddvar6',\
                'ddvar7', 'ddvar8', 'ddvar9', 'ddvar10', 'ddvar11', 'ddvar12','ddskew0', 'ddskew1', 'ddskew2', 'ddskew3', 'ddskew4', 'ddskew5', 'ddskew6',\
                'ddskew7', 'ddskew8', 'ddskew9', 'ddskew10', 'ddskew11', 'ddskew12','ddkurtosis0', 'ddkurtosis1', 'ddkurtosis2', 'ddkurtosis3', 'ddkurtosis4', 'ddkurtosis5', 'ddkurtosis6',\
                'ddkurtosis7', 'ddkurtosis8', 'ddkurtosis9', 'ddkurtosis10', 'ddkurtosis11', 'ddkurtosis12']] = features

            processed = processed.append(pd.DataFrame([filename]))
            # except:
            #     print("Error getting features for " + str(filename))

df.to_csv(main_file, index=False)
processed.to_csv(proc_file, index=False)