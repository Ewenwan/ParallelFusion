#ifndef OPTICAL_FLOW_UTILS_H__
#define OPTICAL_FLOW_UTILS_H__

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

cv::Mat drawFlows(const std::vector<std::pair<double, double> > &flows, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const double MAX_FLOW = 5);
std::vector<std::pair<double, double> > readFlows(const std::string &filename);
void writeFlows(const std::vector<std::pair<double, double> > &flows, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const std::string &filename);
std::vector<std::pair<double, double> > shiftFlows(const std::vector<std::pair<double, double> > &flows, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const double shift_x, const double shift_y);
std::vector<std::pair<double, double> > disturbFlows(const std::vector<std::pair<double, double> > &flows, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const double radius);
double calcFlowsDiff(const std::vector<std::pair<double, double> > &flows_1, const std::vector<std::pair<double, double> > &flows_2, const int IMAGE_WIDTH, const int IMAGE_HEIGHT);
bool onBorder(const int pixel, const int IMAGE_WIDTH, const int IMAGE_HEIGHT);

#endif
