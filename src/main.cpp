#include <iostream>

#include "opencv2/opencv.hpp"    
         
int main(int argc, char** argv) {
	cv::VideoCapture cap;

	if(argc < 2) {
		std::cout << "Webcam mode" << std::endl;
		cap = cv::VideoCapture(0);
	}
	else if(argc < 3) {
		std::cout << "Video file mode" << std::endl;
	}
	else {
		std::cerr << "Usage: best_face_pose [path/to/video_file_name]" << std::endl;
	}

	if(!cap.isOpened())
		return -1;

	cv::Mat frame;
	
	while(cap.grab()) {
		cap.retrieve(frame);
		cv::imshow("Prova", frame);
		cv::waitKey(5);
	}
	return 0;
}
