import shutil 
import os 
import glob

src = r"C:\Users\PC\Documents\School\612\homework-3"

dst = r"C:\Users\PC\Desktop\2\data_to_column"


#Search for pattern
pattern = src + "/*happy.csv"
files = glob.glob(src + pattern)

#Move files with shutil

for file in glob.iglob(pattern, recursive = True): 
  file_name = os.path.basename(file)
  shutil.copyfile(file, dst + file_name)
  print('Moved!', file)