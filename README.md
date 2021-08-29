# vehicle_tracking
### Fast algorithm in OpenCV for vehicle detection with cast shadow removal. 


Algorithm for vehicle detection over the highway - two parallel traffic lines. The work follows some restrictions as the system must be useful during the day time from sunshine till sunset. The functionality must not be affected by outer shadows (trees, street lights, and so forth) and vehicle shadows -- self shadow and cast shadow. The main restriction is in resources limitation to CPU cores only with the least possible requirement for computing capacity.

The system is implemented in Python 3.x and uses the methods from the OpenCV library. The algorithm detects all moving objects and filters the noise from detection to keep only vehicles and their shadow. A vehicle's area and its shadows are further analyzed by feature detection techniques to detect vehicles only.
Present results show good accuracy of vehicle localization in real-time applications.

Repository contains:
* Main algorithm
* Alternative algorithm based on CNN
* 3 records 
  *  result video 
  *  2x test records with masked licence plate by different colors (masked by unpublished algorithm which detects licence plates by contours on record, due to GDPR we cannot publish original streams)
* Background image

![GitHub Logo](/contours_with_detected.png)


