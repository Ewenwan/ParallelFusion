#ifndef OPTICAL_FLOW_UTILS_H__
#define OPTICAL_FLOW_UTILS_H__

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

cv::Mat drawFlows(const std::vector<std::pair<double, double> > &flows, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const double MAX_FLOW = 5);
std::vector<std::pair<double, double> > readFlowsFromFile(const std::string &filename);
std::vector<std::pair<double, double> > shiftFlows(const std::vector<std::pair<double, double> > &flows, const double IMAGE_WIDTH, const double IMAGE_HEIGHT, const double shift_x, const double shift_y);
double calcFlowsDiff(const std::vector<std::pair<double, double> > &flows_1, const std::vector<std::pair<double, double> > &flows_2, const double IMAGE_WIDTH, const double IMAGE_HEIGHT);

#endif
