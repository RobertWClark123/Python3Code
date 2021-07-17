import os
import shutil
import csv

# read the csv file to get a list of all of the images
filenames = []
classids = []
csv_file = 'GTSRB/Final_Test/Images/GT-final_test.csv'
with open(csv_file, newline='') as f:
  reader = csv.DictReader(f, delimiter=';')
  for row in reader:
    filenames.append(row['Filename'])
    classids.append(row['ClassId'])
    
# create an output folder for each class
output_root = 'GTSRD/Test_Arranged'
if not os.path.isdir(output_root):
  os.mkdir(output_root)
unique_class_ids = set(classids)
for clssid in unique_class_ids:
  folder_name = output_root + '/{0:05d}'.format(int(classid))
  if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
    
# copy all of the images into the folders
for (img_name, classid) in zip(filenames, classids):
  full_img_name = 'GTSRB/Final_Test/Images/' + img_name
  folder_name = output_root + '/{0:05d}'.format(int(classid))
  new_img_name = folder_name + '/' + img_name
  
  if not os.path.exists(new_img_name):
  shutil.copyfile(full_img_name, new_img_name)
