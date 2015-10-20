#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xphoto.hpp>

#include <boost/filesystem.hpp>

#include <iostream>
#include <vector>

#include "PassportDetection.h"

using namespace cv;
namespace bfs = boost::filesystem;

int main (int argc, char *argv[])
{
	const char positiveTestPath[] = "../Positive";
	const char negativeTestPath[] = "../Negative";
	const char haarCascadePath[] = "../haarcascades/haarcascade_frontalface_alt2.xml";

	using namespace cv;

	CascadeClassifier cascade;
	if (!cascade.load(haarCascadePath))
	{
		std::cerr << "Error loading cascade: " << haarCascadePath << std::endl;
		return -1;
	}

	if (!bfs::exists(bfs::path(positiveTestPath)))
	{
		std::cerr << "Directory doesn't exist: " << positiveTestPath << std::endl;
		return -1;
	}

	if (!bfs::exists(bfs::path(negativeTestPath)))
	{
		std::cerr << "Directory doesn't exist: " << negativeTestPath << std::endl;
		return -1;
	}

	RunTest(positiveTestPath, cascade, true);

	RunTest(negativeTestPath, cascade, false);
	
	return 0;
}

