# remove audio files with downvotes or no huma metadata for the Mozzila Common Voice Dataset
import pandas as pd
import os

folder_path = 'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-dev'
target_csv = '/cv-valid-dev.csv'
file_folder = '/cv-valid-dev/'

df = pd.read_csv(folder_path + target_csv)

# remove files without speaker information or downvotes
bad_files = df[df["age"].isnull() | df["down_votes"]>0]

for row in bad_files["filename"]:
    fullpath = folder_path+'/'+row
    os.remove(fullpath)

# left join to only keep rows in the original csv file that haven't been removed
good_csv = df.merge(bad_files, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']

# save list of good files to csv
good_csv.to_csv('cv-valid-dev-update.csv', index=False)