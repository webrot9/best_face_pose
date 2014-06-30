#include <iostream>

#include "opencv2/opencv.hpp"

//bool evaluate_pose(cv::Rect face, cv
int main(int argc, char** argv) {
  cv::VideoCapture cap;
  cv::CascadeClassifier frontal_face_cascade;
  cv::CascadeClassifier profile_face_cascade;
  cv::CascadeClassifier eyes_cascade;
  cv::CascadeClassifier mouth_cascade;

  if(argc == 2) {
    std::cout << "Webcam mode" << std::endl;
    cap = cv::VideoCapture(0);
  }
  else if(argc == 3) {
    std::cout << "Video file mode" << std::endl;
    cap = cv::VideoCapture(argv[2]);
  }
  else {
    std::cerr << "Usage: best_face_pose path/to/haarcascade_directory [path/to/video_file_name]" << std::endl;
  }

  std::string haarcascade_dir = std::string(argv[1]);
	
  std::string frontalFaceCascadeFilename = haarcascade_dir + "/haarcascade_frontalface_alt.xml";
  std::string profileFaceCascadeFilename = haarcascade_dir + "/haarcascade_profileface.xml";
  std::string eyeCascadeFilename = haarcascade_dir + "/haarcascade_eye.xml";
  std::string mouthCascadeFilename = haarcascade_dir + "/haarcascade_mcs_mouth.xml";
	
  if(!frontal_face_cascade.load(frontalFaceCascadeFilename) || !profile_face_cascade.load(profileFaceCascadeFilename) || 
     !eyes_cascade.load(eyeCascadeFilename) || !mouth_cascade.load(mouthCascadeFilename)) {
    std::cerr << "Error while loading HAAR cascades." << std::endl;
    return -1;
  }
	
  if(!cap.isOpened())
    return -1;

  cv::Mat frame;
  cv::Mat gray_frame;
	
  while(cap.grab()) {
    cap.retrieve(frame);
		
    // Convert to grayscale
    cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);
		
    // Apply Histogram Equalization
    cv::equalizeHist(gray_frame, gray_frame);
		
    // Detect face, eyes and mouth
    std::vector<cv::Rect> frontal_face;
    std::vector<cv::Rect> profile_face;
    std::vector<cv::Rect> face;
    std::vector<cv::Rect> eyes;
    std::vector<cv::Rect> mouth;
		
    frontal_face_cascade.detectMultiScale(gray_frame, frontal_face, 1.4, 4, 0|CV_HAAR_SCALE_IMAGE, cv::Size(50, 50));
    profile_face_cascade.detectMultiScale(gray_frame, profile_face, 1.4, 4, 0|CV_HAAR_SCALE_IMAGE, cv::Size(50, 50));
		
    face.insert(face.end(), frontal_face.begin(), frontal_face.end());
    face.insert(face.end(), profile_face.begin(), profile_face.end());
		
    int max_area = 0;
    cv::Rect* max_face = NULL;
		
    for(std::vector<cv::Rect>::iterator it = face.begin(); it != face.end(); it++) {
      if(std::max(max_area, it->width * it->height) != max_area) {
	max_face = &(*it);
				
	max_area = it->width * it->height;
      }
    }
	
    if(max_face != NULL) {
      cv::rectangle(frame, *max_face, cv::Scalar(0, 255, 0));
			
      cv::Mat face_image(gray_frame, *max_face);
			
      eyes_cascade.detectMultiScale(face_image, eyes, 1.1, 3, 0, cv::Size(20, 20), cv::Size(50, 50));
			
      cv::Rect good_eye1;
      cv::Rect good_eye2;
      cv::Rect good_mouth;
      good_eye1.y = 0;
      good_eye1.height = 0;
      good_eye2.y = 0;
      good_eye2.height = 0;
      good_mouth.y = 0;
      good_mouth.height = 0;
			
      int best_eye_distance = frame.rows;

      if(eyes.size() >= 2) {
	for(std::vector<cv::Rect>::iterator it = eyes.begin(); it != eyes.end() - 1; it++) {
	  for(std::vector<cv::Rect>::iterator it1 = it + 1; it1 != eyes.end(); it1++) {
	    if(std::abs(it->y - it1->y) < best_eye_distance) {
	      good_eye1 = *it;
	      good_eye2 = *it1;
	    }
	  }
	}
				
	good_eye2.x = good_eye2.x + max_face->x;
	good_eye2.y = good_eye2.y + max_face->y;
      }
      else if(eyes.size() == 1) {
	good_eye1 = eyes[0];
      }
			
      good_eye1.x = good_eye1.x + max_face->x;
      good_eye1.y = good_eye1.y + max_face->y;
			
      int max = std::max(good_eye1.y - max_face->y + good_eye1.height, 
			 good_eye2.y - max_face->y + good_eye2.height);
			
      cv::rectangle(frame, good_eye1, cv::Scalar(0, 0, 255));
      cv::rectangle(frame, good_eye2, cv::Scalar(0, 0, 255));
			
      cv::Rect cropped_face = *max_face;
      cropped_face.y = cropped_face.y + max;
      cropped_face.height = cropped_face.height - max;
      cv::Mat cropped_face_image(gray_frame, cropped_face);

      mouth_cascade.detectMultiScale(cropped_face_image, mouth, 1.1, 5, 0, cv::Size(20,20));
      
      int mouth_position_condition = std::numeric_limits<int>::max();
      int eye_distance = std::abs(good_eye1.x + good_eye1.width/2 - (good_eye2.x + good_eye2.width/2))/2;
      
      for(std::vector<cv::Rect>::iterator it1 = mouth.begin(); it1 != mouth.end(); it1++) {
	it1->x = it1->x + max_face->x;
	it1->y = it1->y + max_face->y + max;
	
				if(eyes.size() >= 2) {
					if(std::abs(eye_distance - (it1->x + it1->width/2)) < mouth_position_condition) {
						mouth_position_condition = std::abs(eye_distance - (it1->x + it1->width/2));
						good_mouth = *it1;
					}
				}
				else {
					if(std::abs(max_face->y + max_face->height - (it1->y + it1->height/2)) < mouth_position_condition) {
						mouth_position_condition = std::abs(max_face->y + max_face->height - (it1->y + it1->height/2));
						good_mouth = *it1;
					}
				}
      }
      
      
      cv::rectangle(frame, good_mouth, cv::Scalar(255, 0, 0));
      
//      evaluate_pose(max_face, good_eye1, good_eye2, 
    }
		
    cv::imshow("Video", frame);
    cv::waitKey(5);
  }
	
  return 0;
}
