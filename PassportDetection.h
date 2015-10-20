#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

using namespace cv;
namespace bfs = boost::filesystem;

const char MainWindowName[] = "Passport Detection";

void DebugImg(const Mat debugImg)
{
	const int maxDimension = std::max(debugImg.size[0], debugImg.size[1]);
	Mat resized;
	const double ratio = double(700) / maxDimension;
	if (ratio != 1)
		resize(debugImg, resized, Size(), ratio, ratio, ratio < 1 ? CV_INTER_AREA : CV_INTER_LINEAR);
	else
		resized = debugImg;
	
	imshow(MainWindowName, debugImg);
	waitKey(0);
	destroyWindow(MainWindowName);
}

enum RotationAngle
{
	Rotate90,
	Rotate180,
	Rotate270,
};

void RotateMatrix(cv::Mat &m, RotationAngle angle)
{
	switch (angle)
	{
	case Rotate90:
		transpose(m, m);  
		flip(m, m, 1);
		break;
	case Rotate180:
		flip(m, m, -1);
		break;
	case Rotate270:
		transpose(m, m);  
		flip(m, m, 0);
		break;
	default:
		throw std::runtime_error("invalid agument 'angle'");
	}
}

Point2f RotatePoint(const Point2f& p, const cv::Mat &m, RotationAngle angle)
{
	switch (angle)
	{
	case Rotate90:
		return Point2f(m.size[0] - p.y, p.x);
	case Rotate180:
		return Point2f(m.size[1] - p.x, m.size[0] - p.y);
	case Rotate270:
		return Point2f(p.y, m.size[1] - p.x);
	default:
		throw std::runtime_error("invalid agument 'angle'");
	}
}

Rect RotateRect(const Rect& r, const cv::Mat &m, RotationAngle angle)
{
	switch (angle)
	{
	case Rotate90:
		return Rect(RotatePoint(r.tl(), m, Rotate90), RotatePoint(r.br(), m, Rotate90));
	case Rotate180:
		return Rect(RotatePoint(r.tl(), m, Rotate180), RotatePoint(r.br(), m, Rotate180));
	case Rotate270:
		return Rect(RotatePoint(r.tl(), m, Rotate270), RotatePoint(r.br(), m, Rotate270));
	default:
		throw std::runtime_error("invalid agument 'angle'");
	}
}

bool CheckPassportCandidate(const Mat testSample, const Rect passportRect, const Rect& faceRect)
{
	if (passportRect.height < passportRect.width)
		return false;

	if ((passportRect & faceRect) != faceRect)
		return false;

	const double areaRatio = passportRect.area() / faceRect.area();
	if (areaRatio < 12 || areaRatio > 110)
	{
		return false;
	}
	
	const Point faceCenterAbs = Point(faceRect.x + faceRect.width / 2, faceRect.y + faceRect.height / 2);
	const Point faceCenterRel = faceCenterAbs - passportRect.tl();
	const double ratio = double(passportRect.width) / 750;

	const int minX = cvRound(137 * ratio - 65 * ratio);
	const int minY = cvRound(730 * ratio - 100 * ratio);
	const int maxX = cvRound(minX + 65 * 2 * ratio);
	const int maxY = cvRound(minY + 100 * 2 * ratio);
	
	return faceCenterRel.x >= minX && faceCenterRel.x <= maxX && faceCenterRel.y >= minY && faceCenterRel.y <= maxY;
}

void DetectRects(const Mat sample, std::vector<Rect>& rects)
{
	Mat blurredSample;
	GaussianBlur(sample, blurredSample, Size(5, 5), 0);
	cv::xphoto::balanceWhite(blurredSample, blurredSample, cv::xphoto::WHITE_BALANCE_SIMPLE);

	Mat edges;
	cv::Canny(blurredSample, edges, 60, 160, 3, true);
	
	std::vector<std::vector<Point>> contours;
	findContours(edges, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); ++i)
	{
		std::vector<Point> approxContour;
		approxPolyDP(contours[i], approxContour, 10, true);
		if (approxContour.size() > 3)
		{
			std::vector<Point> hull;
			convexHull(approxContour, hull);
			const Rect rect = boundingRect(hull);
			const auto minmax = std::minmax(rect.width, rect.height);

			if (minmax.first > 250)
			{
				const double areaRatio = rect.area() / contourArea(hull);
				if (areaRatio > 1.5 || areaRatio < 1)
					continue;

				rects.push_back(rect);
			}
		}
	}
}

bool IsPassport(Mat& sample, CascadeClassifier& cascade, Rect& passportRect, Rect& faceRect)
{
	std::vector<Rect> rects;
	{
		std::vector<Rect> tmp;
		DetectRects(sample,  tmp);
		std::copy_if(tmp.begin(), tmp.end(), std::back_inserter(rects),
			[=](Rect r)
			{
				const auto minmax = std::minmax(r.width, r.height);
				const double aspectRatio = double(minmax.second) / minmax.first;
				return aspectRatio >= 1.15 && aspectRatio <= 1.5;
			}
		);
	}
	rects.push_back(Rect(Point(0, 0), Point(sample.size[1], sample.size[0])));
	
	Size minFaceSize, maxFaceSize;
	{
		const int sampleArea = sample.size[0] * sample.size[1];
		const int minFaceDimension = cvRound(std::sqrt(sampleArea * 0.004));
		minFaceSize = Size(minFaceDimension, minFaceDimension);
		const int maxFaceDimension = cvRound(std::sqrt(sampleArea * 0.048));
		maxFaceSize = Size(maxFaceDimension, maxFaceDimension);
	}

	Mat blurredSample;
	GaussianBlur(sample, blurredSample, Size(5, 7), 0);

	for (int angle = 0; angle < 4; ++angle)
	{
		std::vector<Rect> faceRects;
		cascade.detectMultiScale(blurredSample, faceRects, 1.1, 3, 0, minFaceSize, maxFaceSize);
		
		for (auto fIt = faceRects.begin(); fIt != faceRects.end(); ++fIt)
		{
			for (auto rIt = rects.begin(); rIt != rects.end(); ++rIt)
			{
				if (CheckPassportCandidate(sample, *rIt, *fIt))
				{
					passportRect = *rIt;
					faceRect = *fIt;

					// Uncomment to detected passports
					//Mat debugImg;
					//cvtColor(sample, debugImg, CV_GRAY2BGR);
					//rectangle(debugImg, faceRect, Scalar(0, 255, 0), 2);
					//rectangle(debugImg, passportRect, Scalar(0, 0, 255), 2);
					//DebugImg(debugImg);

					return true;
				}
			}
		}

		std::transform(rects.begin(), rects.end(), rects.begin(),
			[&](Rect& r)
			{
				return RotateRect(r, sample, Rotate90);
			}
		);

		::RotateMatrix(sample, Rotate90);
		::RotateMatrix(blurredSample, Rotate90);
	}

	return false;
}

void RunTest(const char* directoryPath, CascadeClassifier& cascade, bool isPositive)
{
	std::ofstream log;
	if (isPositive)
		log.open("positive.log", std::ios_base::trunc);
	else
		log.open("negative.log", std::ios_base::trunc);

	size_t filesCounter = 0;
	size_t detectCounter = 0;

	std::cout << "Inside directory: " << directoryPath << std::endl;
	time_t startTime = time(0);
	bfs::path dirPath(directoryPath);
	for (bfs::recursive_directory_iterator dirContentIt(dirPath), end; dirContentIt != end; ++dirContentIt)
	{
		const auto& path = dirContentIt->path();
		if (!bfs::is_regular_file(path))
			continue;

		const auto extension = bfs::extension(path);
		if (extension != ".png" && extension != ".jpg" && extension != ".jpeg" && extension != ".tif")
			continue;

		Mat sample = imread(path.string(), CV_LOAD_IMAGE_GRAYSCALE);
		if(!sample.data)
		{
			std::cerr << "Error loading sample: " << path << std::endl;
			continue;
		}

		const int maxDimension = std::max(sample.size[0], sample.size[1]);
		if (maxDimension != 1000)
		{
			const double ratio = double(1000) / maxDimension;
			if (ratio != 1)
				resize(sample, sample, Size(), ratio, ratio, ratio < 1 ? CV_INTER_AREA : CV_INTER_LINEAR);
		}

		++filesCounter;
		
		Rect passportRect;
		Rect faceRect;
		if (IsPassport(sample, cascade, passportRect, faceRect))
		{
			if (!isPositive)
				log << path.string() << std::endl;
			++detectCounter;
		}
		else
		{
			if (isPositive)
				log << path.string() << std::endl;
		}
		
		if (filesCounter % 10 == 0)
			std::cout << filesCounter << " files processed..." << std::endl;
	}

	if (filesCounter != 0)
	{
		const time_t timeElapsed = time(0) - startTime;
		std::cout.precision(3);
		const double detectionRate = double(detectCounter) / filesCounter * 100;
		const double speed = double(filesCounter) / timeElapsed;
		log << detectionRate << std::endl << speed << std::endl;
		if (isPositive)
		{
			std::cout << "[Positive Test Results]" << std::endl 
				<< "Detection rate: " << detectCounter << "/" << filesCounter << ", " << detectionRate << "%" << std::endl
				<< "Speed: " << filesCounter << "/" << timeElapsed << ", " << speed << " img/sec" << std::endl << std::endl;
		}
		else
		{
			std::cout << "[Negative Test Results]" << std::endl 
				<< "False alarm rate: " << detectCounter << "/" << filesCounter << ", " << detectionRate << "%" << std::endl
				<< "Speed: " << filesCounter << "/" << timeElapsed << ", " << speed << " img/sec" << std::endl << std::endl;
		}
	}
	else
	{
		std::cerr << "No samples found" << std::endl;
	}
}