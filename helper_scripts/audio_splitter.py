# Split audio into 1 second segments
# Create CSV file 
import csv
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
from csv import writer, reader

# where non-chunked files are stored
input_folder = 'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/RawFiles/JoinedSpeech'
#'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-dev/cv-valid-dev-mix'
#'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/RawFiles'
#'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-train/cv-valid-train-mix'
# output directory of chunked files
output_folder = 'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/RawFiles/SpeechChunks'
#'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-dev/cv-dev-mix-chunks'
#'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/Chunks'
#'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-train/cv-train-mix-chunks'
#
main_csv = 'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-train/cv_train_main_file.csv'
#'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-dev/cv_val_main_file.csv'
#'main_file.csv'
#'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/CV-files/cv-valid-train/cv_train_main_file.csv'
chunk_length = 1000 # chunk lenght in ms

## read csv file to get list of files that have already been chunked
csv_file = open(main_csv, 'r')
csv_reader = csv.reader(csv_file, delimiter=',')
processed_files = set() # get unique filenames 
for row in csv_reader:
    processed_files.add(row[0]) # column with orginal file names (2 for personal, 0 for cv) !!!!! CHANGE
csv_file.close()
#print(processed_files)
## reopen csv file to append filenames
csv_file = open(main_csv, 'a', newline='')
write_csv = writer(csv_file)

# go through all files in input folder
for (dirpath, dirnames, filenames) in os.walk(input_folder):
    for filename in filenames:
        if filename.endswith('.wav') and filename not in processed_files:
            filepath = dirpath + '/' + filename
            filename_noext = filename[:-4]
            try:
                thisAudio = AudioSegment.from_file(filepath, "wav")
                chunks = make_chunks(thisAudio, chunk_length)
                for i, chunk in enumerate(chunks):
                    if chunk.duration_seconds >=1:
                        chunk_name = str(filename_noext)+" - {0}.wav".format(i)
                        chunk_path = output_folder + '/' + chunk_name
                        chunk.export(chunk_path, format="wav")
                        new_line = [filename, chunk_name]
                        #new_line = [filename_noext, chunk_name, filename]
                        write_csv.writerow(new_line)
                print("Created {0} ms chunks for ".format(chunk_length)+ str(filename))
            except:
                print("ERROR CHUNKING " + str(filename))

csv_file.close()