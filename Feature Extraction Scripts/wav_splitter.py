import sys
import opt
import math
from pydub import AudioSegment

# Name and directory of the WAV file to be split go here
directory = 'C:/Language_Audio_Samples/'
filename = 'german_male.wav'

class WAV_Splitter():
    def __init__(self, directory, filename):
        self.directory = directory; self.filename = filename; self.filepath = directory + '\\' + filename
        self.audio = AudioSegment.from_wav(self.filepath)
    
    # Gets the length of the WAV file
    def get_WAV_length(self):
        return self.audio.duration_seconds
    
    # Gets a single 10 second clip within the specified time range
    def get_10sec_split(self, start_time, end_time, split_filename):
        t0 = start_time * 10 * 1000
        t1 = end_time * 10 * 1000
        
        split_audio = self.audio[t0:t1]
        split_audio.export(self.directory + '\\' + split_filename, format="wav")
        
    # Splits the requested file into 10 second clips
    def split_WAV(self, sec_per_split):
        print('Splitting ' + filename + ' into 10 second clips')
        
        # Find the number of splits to be made
        total_splits = math.ceil(self.get_WAV_length() / 10)
        
        # Perform the split
        for i in range(0, total_splits, sec_per_split):
            split_fn = str(i) + '_' + self.filename
            self.get_10sec_split(i, i + sec_per_split, split_fn)
            
            print('Split ' + str(i) + ' Complete')
            if i == total_splits - sec_per_split:
                print('Split Complete')


# Initialise the splitter
split_wav = WAV_Splitter(directory, filename)

# Perform the split
split_wav.split_WAV(sec_per_split=1)