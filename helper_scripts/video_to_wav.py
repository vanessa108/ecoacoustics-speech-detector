import moviepy.editor as mp
import os
# convers mp4 to WAV files
folder = 'C:/Users/vanes/OneDrive - Queensland University of Technology/Sem 2 2021/EGH400-2/RawFiles'
formats_to_convert = ['.mp4']
for (dirpath, dirnames, filenames) in os.walk(folder):
    for filename in filenames:
        if filename.endswith(tuple(formats_to_convert)):
            filepath = dirpath + '/' + filename
            (path, file_extension) = os.path.splitext(filepath)
            file_extension_final = file_extension.replace('.', '')
            try:
                wav_filename = filename.replace(file_extension_final, 'wav')
                wav_path = dirpath + '/' + wav_filename
                clip = mp.VideoFileClip(filepath)
                clip.audio.write_audiofile(wav_path)
                clip.close()
                os.remove(filepath)
            except:
                print("ERROR CONVERTING " + str(filepath))