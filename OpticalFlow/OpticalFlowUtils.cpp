#include "OpticalFlowUtils.h"

#include <fstream>

#include "../base/cv_utils/cv_utils.h"

using namespace std;
using namespace cv;

namespace flow_fusion {
    cv::Mat drawFlows(const std::vector<std::pair<double, double> > &flows, const int IMAGE_WIDTH,
                      const int IMAGE_HEIGHT, const double MAX_FLOW) {
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

    std::vector<std::pair<double, double> > readFlows(const std::string &filename) {
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
      flow_in_str.close();
      vector<pair<double, double> > flows(image_width * image_height);
      for (int pixel = 0; pixel < image_width * image_height; pixel++) {
        flows[pixel].first = flow_values_float[pixel * 2 + 0];
        flows[pixel].second = flow_values_float[pixel * 2 + 1];
      }
      return flows;
    }

    void writeFlows(const std::vector<std::pair<double, double> > &flows, const int IMAGE_WIDTH, const int IMAGE_HEIGHT,
                    const std::string &filename) {
      ofstream flow_out_str(filename, ios::binary);
      if (!flow_out_str)
        exit(1);
      float tag = 0;
      int image_width_temp = IMAGE_WIDTH;
      int image_height_temp = IMAGE_HEIGHT;
      flow_out_str.write(reinterpret_cast<char *>(&tag), sizeof(float));
      flow_out_str.write(reinterpret_cast<char *>(&image_width_temp), sizeof(int));
      flow_out_str.write(reinterpret_cast<char *>(&image_height_temp), sizeof(int));
      vector<float> flow_values_float(IMAGE_WIDTH * IMAGE_HEIGHT * 2);
      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
        flow_values_float[pixel * 2 + 0] = flows[pixel].first;
        flow_values_float[pixel * 2 + 1] = flows[pixel].second;
      }

      flow_out_str.write(reinterpret_cast<char *>(&flow_values_float[0]),
                         sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT * 2);
      flow_out_str.close();
    }

    std::vector<std::pair<double, double> > shiftFlows(const std::vector<std::pair<double, double> > &flows,
                                                       const int IMAGE_WIDTH, const int IMAGE_HEIGHT,
                                                       const double shift_x, const double shift_y) {
      vector<pair<double, double> > shifted_flows(IMAGE_WIDTH * IMAGE_HEIGHT);
      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
        shifted_flows[pixel] = make_pair(flows[pixel].first + shift_x, flows[pixel].second + shift_y);
      return shifted_flows;
    }

    std::vector<std::pair<double, double> > disturbFlows(const std::vector<std::pair<double, double> > &flows,
                                                         const int IMAGE_WIDTH, const int IMAGE_HEIGHT,
                                                         const double radius) {
      vector<pair<double, double> > disturbed_flows(IMAGE_WIDTH * IMAGE_HEIGHT);
      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
        disturbed_flows[pixel] = make_pair(flows[pixel].first + cv_utils::drawFromNormalDistribution(0, radius),
                                           flows[pixel].second + cv_utils::drawFromNormalDistribution(0, radius));
      return disturbed_flows;
    }

    double calcFlowsDiff(const std::vector<std::pair<double, double> > &flows_1,
                         const std::vector<std::pair<double, double> > &flows_2, const int IMAGE_WIDTH,
                         const int IMAGE_HEIGHT) {
      double flow_diff2_sum = 0;
      int num_valid_pixels = 0;
      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
        if (onBorder(pixel, IMAGE_WIDTH, IMAGE_HEIGHT))
          continue;
        if (abs(flows_1[pixel].first) > IMAGE_WIDTH || abs(flows_1[pixel].second) > IMAGE_HEIGHT ||
            abs(flows_2[pixel].first) > IMAGE_WIDTH || abs(flows_2[pixel].second) > IMAGE_HEIGHT)
          continue;
        flow_diff2_sum += pow(flows_1[pixel].first - flows_2[pixel].first, 2) +
                          pow(flows_1[pixel].second - flows_2[pixel].second, 2);
        num_valid_pixels++;
      }
      double flow_diff = sqrt(flow_diff2_sum / num_valid_pixels);
      return flow_diff;
    }

    bool onBorder(const int pixel, const int IMAGE_WIDTH, const int IMAGE_HEIGHT) {
      const int IMAGE_BORDER = 10;
      if (pixel % IMAGE_WIDTH < IMAGE_BORDER || pixel % IMAGE_WIDTH > IMAGE_WIDTH - 1 - IMAGE_BORDER ||
          pixel / IMAGE_WIDTH < IMAGE_BORDER || pixel / IMAGE_WIDTH > IMAGE_HEIGHT - 1 - IMAGE_BORDER)
        return true;
      else
        return false;
    }
}