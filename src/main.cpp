#include <iostream>

#include "opencv2/opencv.hpp"


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
		frontal_face_cascade.detectMultiScale(gray_frame, frontal_face, 1.5, 1, 0|CV_HAAR_SCALE_IMAGE, cv::Size(60, 60));
		profile_face_cascade.detectMultiScale(gray_frame, profile_face, 1.5, 1, 0|CV_HAAR_SCALE_IMAGE, cv::Size(60, 60));
		
		face.insert(face.end(), frontal_face.begin(), frontal_face.end());
		face.insert(face.end(), profile_face.begin(), profile_face.end());
		
		eyes_cascade.detectMultiScale(gray_frame, eyes, 1.2, 3, 0, cv::Size(20, 20), cv::Size(50, 50));
		mouth_cascade.detectMultiScale(gray_frame, mouth, 1.2, 4, 0, cv::Size(30,50));

		for(std::vector<cv::Rect>::iterator it = face.begin(); it != face.end(); it++)
			cv::rectangle(frame, *it, cv::Scalar(0, 255, 0));
		
		for(std::vector<cv::Rect>::iterator it = eyes.begin(); it != eyes.end(); it++)
			cv::rectangle(frame, *it, cv::Scalar(0, 0, 255));
		
		for(std::vector<cv::Rect>::iterator it = mouth.begin(); it != mouth.end(); it++)
			cv::rectangle(frame, *it, cv::Scalar(255, 0, 0));
		
		cv::imshow("Video", frame);
		cv::waitKey(5);
	}
	
	return 0;
}
