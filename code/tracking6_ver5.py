import numpy as np
import cv2
import globalVariables as gV
import random
import pickle


gV.selRoi = 0
gV.first_time = 1
frame_refresh = gV.number_of_frame_tracking
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#load bounding box info
with open('ids_in_frame.pickle', 'rb') as file:
    ids_in_frame = pickle.load(file)
bb_matrix = np.load('box_matrix.npy')



def findDistance(r1,c1,r2,c2):
	d = (r1-r2)**2 + (c1-c2)**2
	d = d**0.5
	return d
		
#main function


cap = cv2.VideoCapture('../Dataset/Segment_30min_Day1_CAM12_20fps_960x540.mp4')
# cap.set(1,1800)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#videoWriter = cv2.VideoWriter('oto_other.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

cv2.namedWindow('tracker')


all_person = set([])
for list_ in ids_in_frame:
    all_person = all_person | set(list_)

person_corners_dict = {}
person_corners_dict_new = {}

for person_id in all_person :
	person_corners_dict[person_id] = []
	person_corners_dict_new[person_id] = []

count = 0

new_corners = [[0,0]]
# iterate = 0
while True:
	while True:
		# print(count)
		_,frame = cap.read() 
		for ids in all_person:
		# for ids in ids_in_frame[count]:
			if ids in ids_in_frame[count]:
				tid = int(ids)-1
				gV.top_left = [int(bb_matrix[tid][count][1]) ,int(bb_matrix[tid][count][0]) ]
				gV.bottom_right = [int(bb_matrix[tid][count][3]) ,int(bb_matrix[tid][count][2]) ]
				#-----Drawing Stuff on the Image
				cv2.rectangle(frame,(gV.top_left[1],gV.top_left[0]),(gV.bottom_right[1],gV.bottom_right[0]),color = (255,0,0),thickness = 2)

				#-----Finding ROI and extracting Corners
				frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
				roi = frameGray[gV.top_left[0]:gV.bottom_right[0], gV.top_left[1]:gV.bottom_right[1]  ] #selecting roi
				new_corners = cv2.goodFeaturesToTrack(roi,50,0.01,10) #find corners
				
				#-----converting to complete image coordinates (new_corners)
				# if new_corners != None:
				new_corners[:,0,0] = new_corners[:,0,0] + gV.top_left[1]
				new_corners[:,0,1] = new_corners[:,0,1] + gV.top_left[0]

				person_corners_dict[ids].append(new_corners)

				#-----drawing the corners in the original image
				for corner in new_corners:
					cv2.circle(frame, (int(corner[0][0]),int(corner[0][1])) ,2,(0,0,255),2)

				#-----old_corners and oldFrame is updated
				oldFrameGray = frameGray.copy()
				old_corners = new_corners.copy()
		
				cv2.imshow('tracker',frame)
			else:
				person_corners_dict[ids].append(np.zeros_like(new_corners))

		count += 1
		# a = cv2.waitKey(5)
		# if a == 27:
		# 	cv2.destroyAllWindows()
		# 	cap.release()
		# 	videoWriter.release
		# elif a == 97:
		# 	break
		break
	total_shift = {}

	first_ = True
	#----Actual Tracking-----
	while True:
		'Now we have oldFrame,we can get new_frame,we have old corners and we can get new corners and update accordingly'
	
		#read new frame and cvt to gray
		ret,frame = cap.read()
		frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		for ids in ids_in_frame[count]:
			if first_ == True :
				old_corners = person_corners_dict[ids][int(count/frame_refresh)]
				total_shift[ids] = 0
			else:
				old_corners = person_corners_dict_new[ids] 

			if len(old_corners) == 0:
				break
		#finding the new tracked points
			new_corners, st, err = cv2.calcOpticalFlowPyrLK(oldFrameGray, frameGray, old_corners, None, **lk_params)
			# print(ids,np.sqrt(np.sum((np.array(new_corners)-np.array(old_corners))**2,axis = 1)))

			delete_lists = np.where(np.sqrt(np.sum((np.array(new_corners)-np.array(old_corners))**2,axis = 1))<= 0.1)[0].tolist()
			cal_old = np.delete(old_corners,delete_lists,0)
			cal_new = np.delete(new_corners,delete_lists,0)

			try:
				total_shift[ids] += np.sum(np.sqrt(np.sum((np.array(cal_old)-np.array(cal_new))**2,axis = 1)))
			except:
				total_shift[ids] = 0


			#---pruning far away points:
			#first finding centroid
			r_add,c_add = 0,0
			for corner in new_corners:
				r_add = r_add + corner[0][1]
				c_add = c_add + corner[0][0]
			centroid_row = int(1.0*r_add/len(new_corners))
			centroid_col = int(1.0*c_add/len(new_corners))
			#draw centroid
			cv2.circle(frame,(int(centroid_col),int(centroid_row)),2,(255,0,0)) 
			#add only those corners to new_corners_updated which are at a distance of 30 or less
			new_corners_updated = new_corners.copy()
			tobedel = []

			for index in range(len(new_corners)):
				if findDistance(new_corners[index][0][1],new_corners[index][0][0],int(centroid_row),int(centroid_col)) > 200:
					tobedel.append(index)
			new_corners_updated = np.delete(new_corners_updated,tobedel,0)

			# new_corners_plot = np.setxor1d(new_corners_updated, cal_new)

			#drawing the new points
			for corner in new_corners_updated:
				cv2.circle(frame, (int(corner[0][0]),int(corner[0][1])) ,2,(0,0,255))
			if len(new_corners_updated) < 1:
				print ('OBJECT LOST, Reinitialize for tracking')
				break
			#finding the min enclosing circle
			ctr , rad = cv2.minEnclosingCircle(new_corners_updated)
		
			cv2.circle(frame, (int(ctr[0]),int(ctr[1])) ,int(rad),(0,0,255),thickness = 5)	
			
			person_corners_dict_new[ids] = new_corners_updated


		#updating old_corners and oldFrameGray 
		oldFrameGray = frameGray.copy()
		# old_corners = new_corners_updated.copy()
	
		#showing stuff on video
		# cv2.putText(frame,'Tracking Integrity : Excellent %04.3f'%random.random(),(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,color = (200,50,75),thickness = 3)
		cv2.imshow('tracker',frame)
		first_ = False
		
		count += 1

		a = cv2.waitKey(5)
		if a== 27:
			cv2.destroyAllWindows()
			cap.release()
		elif a == 97 or (count!= 0 and count%frame_refresh==0):
			print("distance of id:'2'",total_shift['2'])
			break
		
cv2.destroyAllWindows()		
	
		
		
		
	
