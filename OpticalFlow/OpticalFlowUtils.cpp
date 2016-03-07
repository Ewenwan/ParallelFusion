#include "OpticalFlowUtils.h"

#include <fstream>

using namespace std;
using namespace cv;

cv::Mat drawFlows(const std::vector<std::pair<double, double> > &flows, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const double MAX_FLOW)
{
  Mat flow_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
  
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
    double V = 0.8;
    double S = min(sqrt(pow(flows[pixel].first, 2) + pow(flows[pixel].second, 2)) / MAX_FLOW, 1.0);
    double H = atan2(flows[pixel].second, flows[pixel].first);
    //cout << H << '\t' << S << endl;
    double F = H - floor(H / (60 * M_PI / 180));
    double P = V * (1 - S);
    double Q = V * (1 - S * F);
    double T = V * (1 - S * (1 - F));
    //double C = V * S;
    // double X = C * (1 - abs(static_cast<int>(H / (60 * M_PI / 180)) % 2 - 1));
    // double m = V - C;
    switch (static_cast<int>((H + 2 * M_PI) / (60 * M_PI / 180)) % 6) {
    case 0:
      flow_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = Vec3b(P * 255, T * 255, V * 255);
      break;
    case 1:
      flow_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = Vec3b(P * 255, V * 255, Q * 255);
      break;
    case 2:
      flow_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = Vec3b(T * 255, V * 255, P * 255);
      break;
    case 3:
      flow_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = Vec3b(V * 255, Q * 255, P * 255);
      break;
    case 4:
      flow_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = Vec3b(V * 255, P * 255, T * 255);
      break;
    case 5:
      flow_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = Vec3b(Q * 255, P * 255, V * 255);
      break;
    }
    //cout << flow_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) << endl;
  }
  return flow_image;
}

std::vector<std::pair<double, double> > readFlowsFromFile(const std::string &filename)
{  
  ifstream flow_in_str(filename, ios::binary);
  if (!flow_in_str)
    exit(1);
  int image_width, image_height;
  float tag;
  flow_in_str.read(reinterpret_cast<char *>(&tag), sizeof(float));
  flow_in_str.read(reinterpret_cast<char *>(&image_width), sizeof(int));
  flow_in_str.read(reinterpret_cast<char *>(&image_height), sizeof(int));
  vector<float> flow_values_float(image_width * image_height * 2);
  flow_in_str.read(reinterpret_cast<char *>(&flow_values_float[0]), sizeof(float) * image_width * image_height * 2);
  vector<pair<double, double> > flows(image_width * image_height);
  for (int pixel = 0; pixel < image_width * image_height; pixel++) {
    flows[pixel].first = flow_values_float[pixel * 2 + 0];
    flows[pixel].second = flow_values_float[pixel * 2 + 1];
  }
  return flows;
}

std::vector<std::pair<double, double> > shiftFlows(const std::vector<std::pair<double, double> > &flows, const double IMAGE_WIDTH, const double IMAGE_HEIGHT, const double shift_x, const double shift_y)
{
  vector<pair<double, double> > shifted_flows(IMAGE_WIDTH * IMAGE_HEIGHT);
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
    shifted_flows[pixel] = make_pair(flows[pixel].first + shift_x, flows[pixel].second + shift_y);
  return shifted_flows;
}

double calcFlowsDiff(const std::vector<std::pair<double, double> > &flows_1, const std::vector<std::pair<double, double> > &flows_2, const double IMAGE_WIDTH, const double IMAGE_HEIGHT)
{
  double flow_diff = 0;
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
    flow_diff += pow(flows_1[pixel].first - flows_2[pixel].first, 2) + pow(flows_1[pixel].second - flows_2[pixel].second, 2);
  flow_diff = sqrt(flow_diff);
  return flow_diff;
}
