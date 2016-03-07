#include "OpticalFlowCalculation.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaoptflow.hpp>

#include "lib/OpticalFlow/OpticalFlow.h"
#include "lib/OpticalFlow/GaussianPyramid.h"

using namespace std;
using namespace cv;

vector<pair<double, double> > calcFlowsPyrLK(const Mat &image_1, const Mat &image_2, const int NUM_LEVELS)
{
  const int IMAGE_WIDTH = image_1.cols;
  const int IMAGE_HEIGHT = image_1.rows;
  Mat points_1(1, IMAGE_WIDTH * IMAGE_HEIGHT, CV_32FC2);
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
    points_1.at<Vec2f>(0, pixel) = Vec2f(pixel % IMAGE_WIDTH, pixel / IMAGE_WIDTH);

  Mat points_2;
  Mat status;
  Mat err;
  calcOpticalFlowPyrLK(image_1, image_2, points_1, points_2, status, err, Size(21, 21), NUM_LEVELS);

  vector<pair<double, double> > flows(IMAGE_WIDTH * IMAGE_HEIGHT);
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
    Vec2f point_1 = points_1.at<Vec2f>(0, pixel);
    Vec2f point_2 = points_2.at<Vec2f>(0, pixel);
    flows[pixel].first = point_2[0] - point_1[0];
    flows[pixel].second = point_2[1] - point_1[1];
  }
  return flows;
}

std::vector<std::pair<double, double> > calcFlowsFarneback(const cv::Mat &image_1, const cv::Mat &image_2, const int NUM_LEVELS, const int POLY_N, const int FLAGS)
{
  const double PYR_SCALE = 0.75;
  const int WINDOW_SIZE = 15;
  const int NUM_ITERATIONS = 5;
  //const double POLY_SIGMA = 0.3;
  const double POLY_SIGMA = 0.3 + 0.2 * (POLY_N - 1);
  
  const int IMAGE_WIDTH = image_1.cols;
  const int IMAGE_HEIGHT = image_1.rows;

  Mat image_1_gray;
  if (image_1.channels() == 1)
    image_1_gray = image_1.clone();
  else
    cvtColor(image_1, image_1_gray, CV_BGR2GRAY);
  Mat image_2_gray;
  if (image_2.channels() == 1)
    image_2_gray = image_2.clone();
  else
    cvtColor(image_2, image_2_gray, CV_BGR2GRAY);
  
  
  Mat flow_image;
  calcOpticalFlowFarneback(image_1_gray, image_2_gray, flow_image, PYR_SCALE, NUM_LEVELS, WINDOW_SIZE, NUM_ITERATIONS, POLY_N, POLY_SIGMA, FLAGS);

  vector<pair<double, double> > flows(IMAGE_WIDTH * IMAGE_HEIGHT);
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
    Vec2f flow = flow_image.at<Vec2f>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH);
    flows[pixel].first = flow[0];
      flows[pixel].second = flow[1];
  }
  return flows;
}

std::vector<std::pair<double, double> > calcFlowsLayerWise(const cv::Mat &image_1, const cv::Mat &image_2, const double ALPHA, const double RATIO)
{
  const int IMAGE_WIDTH = image_1.cols;
  const int IMAGE_HEIGHT = image_1.rows;

  // Mat image_1_small, image_2_small;
  // downSample(img1, img1_small, nLevel);
  // downSample(img2, img2_small, nLevel);
  DImage d_image_1, d_image_2, d_flow_image;
  //copy image data to DImage
  d_image_1.allocate(image_1.cols, image_1.rows, image_1.channels());
  d_image_2.allocate(image_2.cols, image_2.rows, image_2.channels());
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
    Vec3b color_1 = image_1.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH);
    for (int c = 0; c < 3; c++)
      d_image_1[pixel * 3 + c] = 1.0 * color_1[c] / 255;
    Vec3b color_2 = image_2.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH);
    for (int c = 0; c < 3; c++)
      d_image_2[pixel * 3 + c] = 1.0 * color_2[c] / 255;
  }
  
  d_flow_image.allocate(d_image_1.width(), d_image_1.height(), 2);

  const int MIN_WIDTH = 30;
  const int NUM_OUTER_FP_ITERATIONS = 10;
  const int NUM_INNER_FP_ITERATIONS = 1;
  const int NUM_CG_ITERATIONS = 30;

  DImage weight_image;
  weight_image.allocate(d_image_1.width(), d_image_1.height(), 1);
  weight_image.setValue(1);

  DImage vx, vy, warped_image;
  OpticalFlow::Coarse2FineFlow(vx, vy, warped_image, d_image_1, d_image_2, weight_image, ALPHA, RATIO, MIN_WIDTH, NUM_OUTER_FP_ITERATIONS, NUM_INNER_FP_ITERATIONS, NUM_CG_ITERATIONS);

  vector<pair<double, double> > flows(IMAGE_WIDTH * IMAGE_HEIGHT);
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
    flows[pixel].first = vx[pixel];
    flows[pixel].second = vy[pixel];
  }
  return flows;
}

std::vector<std::pair<double, double> > calcFlowsBrox(const cv::Mat &image_1, const cv::Mat &image_2, const double ALPHA, const double RATIO)
{
  const int IMAGE_WIDTH = image_1.cols;
  const int IMAGE_HEIGHT = image_1.rows;

  cv::Ptr<cv::cuda::BroxOpticalFlow>  brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
  
  cuda::GpuMat image_1_GPU(image_1);
  cuda::GpuMat image_2_GPU(image_2);
  cuda::GpuMat flow_image_GPU(image_1.size(), CV_32FC2);
  cuda::GpuMat image_1_GPU_float, image_2_GPU_float;
  image_1_GPU.convertTo(image_1_GPU_float, CV_32F, 1.0 / 255.0);
  image_2_GPU.convertTo(image_2_GPU_float, CV_32F, 1.0 / 255.0);
  brox->calc(image_1_GPU_float, image_2_GPU_float, flow_image_GPU);
  Mat flow_image_CPU(flow_image_GPU.clone());

  vector<pair<double, double> > flows(IMAGE_WIDTH * IMAGE_HEIGHT);
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
    Vec2f flow = flow_image_CPU.at<Vec2f>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH);   
    flows[pixel].first = flow[0];
    flows[pixel].second = flow[1];
  }
  return flows;
}
