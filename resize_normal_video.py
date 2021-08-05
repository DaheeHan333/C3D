import os
import glob
import cv2

dirpath = '/home/proj_vode/normal_dataset/'

i = 'Training-Normal-Videos-Part-2'

if os.path.exists('/home/proj_vode/jpg_dataset/' + i) == False :
    os.mkdir('/home/proj_vode/jpg_dataset/' + i)

dirpath2 = dirpath + '/' + i
filepath2 = os.listdir(dirpath2) #237-1 ...
count = 483

for j in filepath2:
    if count == 503 : break

    if 'zip' in j or 'empty' in j:
        continue
    mp4path = glob.glob(dirpath2 + '/' + j + '/*.mp4')
    
    print(mp4path)
    
    
    for mp4 in mp4path:
        video = mp4

        vs = cv2.VideoCapture(video)

        writer = None

        if os.path.exists('/home/proj_vode/jpg_dataset/' + 'Normal' + '/' + 'Normal' + str(count).zfill(3) + '_x264.mp4') == False :
            os.mkdir('/home/proj_vode/jpg_dataset/' + 'Normal' + '/' + 'Normal' + str(count).zfill(3) + '_x264.mp4')
        
        subcount = 1
        while True:
            ret, frame = vs.read()

            if frame is None:
                break
            
            frame2 = cv2.resize(frame, (224, 224))

            cv2.imwrite('/home/proj_vode/jpg_dataset/' + 'Normal' + '/' + 'Normal' + str(count).zfill(3) + '_x264.mp4' + '/' + i+ str(subcount).zfill(5) + '.jpg', frame2)
            subcount += 1
        count += 1

        print('==================================', mp4)

        vs.release()

