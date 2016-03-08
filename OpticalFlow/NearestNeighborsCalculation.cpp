#include "OpticalFlowCalculation.h"

#include <iostream>

using namespace std;
using namespace cv;

namespace
{
  vector<int> calcCommonWindowOffsets(const int pixel_1, const int pixel_2, const int image_width, const int image_height, const int WINDOW_SIZE)
  {
    vector<int> window_offsets;
    for (int offset_x = -(WINDOW_SIZE - 1) / 2; offset_x <= (WINDOW_SIZE - 1) / 2; offset_x++)
      for (int offset_y = -(WINDOW_SIZE - 1) / 2; offset_y <= (WINDOW_SIZE - 1) / 2; offset_y++)
        if (pixel_1 % image_width + offset_x >= 0 && pixel_1 % image_width + offset_x < image_width && pixel_1 / image_width + offset_y >= 0 && pixel_1 / image_width + offset_y < image_height
            && pixel_2 % image_width + offset_x >= 0 && pixel_2 % image_width + offset_x < image_width && pixel_2 / image_width + offset_y >= 0 && pixel_2 / image_width + offset_y < image_height)
          window_offsets.push_back((offset_y + (WINDOW_SIZE - 1) / 2) * WINDOW_SIZE + (offset_x + (WINDOW_SIZE - 1) / 2));
    return window_offsets;
  }
  
  double calcPatchDistance(const Mat &source_image, const Mat &target_image, const int source_pixel, const int target_pixel, const int WINDOW_SIZE)
  {
    const int IMAGE_WIDTH = source_image.cols;
    const int IMAGE_HEIGHT = source_image.rows;
    const double CONFIDENT_PIXEL_WEIGHT = 100;
    
    vector<int> common_window_offsets = calcCommonWindowOffsets(source_pixel, target_pixel, IMAGE_WIDTH, IMAGE_HEIGHT, WINDOW_SIZE);
    double SSD = 0;
    double sum_confidence = 0;
    vector<bool> used_offsets(WINDOW_SIZE * WINDOW_SIZE, false);
    for (vector<int>::const_iterator window_offset_it = common_window_offsets.begin(); window_offset_it != common_window_offsets.end(); window_offset_it++) {
      used_offsets[*window_offset_it] = true;
      int offset_x = *window_offset_it % WINDOW_SIZE - (WINDOW_SIZE - 1) / 2;
      int offset_y = *window_offset_it / WINDOW_SIZE - (WINDOW_SIZE - 1) / 2;
      Vec3b color_1 = source_image.at<Vec3b>(source_pixel / IMAGE_WIDTH + offset_y, source_pixel % IMAGE_WIDTH + offset_x);
      Vec3b color_2 = target_image.at<Vec3b>(target_pixel / IMAGE_WIDTH + offset_y, target_pixel % IMAGE_WIDTH + offset_x);
      
      double confidence = 1; //pow(1.3, -source_distance_map[(target_pixel / IMAGE_WIDTH + offset_y) * IMAGE_WIDTH + target_pixel % IMAGE_WIDTH + offset_x]);
      //double confidence = source_mask.at((target_pixel / IMAGE_WIDTH + offset_y) * IMAGE_WIDTH + target_pixel % IMAGE_WIDTH + offset_x) ? CONFIDENT_PIXEL_WEIGHT : 1;
      
      for (int c = 0; c < 3; c++)
        SSD += pow(1.0 * (color_1[c] - color_2[c]) / 255, 2) * confidence;
      sum_confidence += confidence;
    }
    
    vector<int> target_window_offsets = calcCommonWindowOffsets(target_pixel, target_pixel, IMAGE_WIDTH, IMAGE_HEIGHT, WINDOW_SIZE);
    SSD += 3 * (target_window_offsets.size() - common_window_offsets.size());
    sum_confidence += (target_window_offsets.size() - common_window_offsets.size());
    
    if (sum_confidence == 0)
      return 1;
    
    double distance = SSD / (sum_confidence * 3);
    return distance;
  }
  
  void findBetterNearestNeighbor(const Mat &source_image, const Mat &target_image, vector<int> &nearest_neighbor_field, vector<double> &distance_field, const int pixel, const int direction, const int WINDOW_SIZE)
  {
    const int IMAGE_WIDTH = source_image.cols;
    const int IMAGE_HEIGHT = source_image.rows;
    
    int x = pixel % IMAGE_WIDTH;
    int y = pixel / IMAGE_WIDTH;
    
    int current_nearest_neighbor = nearest_neighbor_field[pixel];
    int current_nearest_neighbor_x = current_nearest_neighbor % IMAGE_WIDTH;
    int current_nearest_neighbor_y = current_nearest_neighbor / IMAGE_WIDTH;
    double current_distance = distance_field[pixel];
    
    int best_nearest_neighbor = current_nearest_neighbor;
    double min_distance = current_distance;
    
    if (x + direction >= 0 && x + direction < IMAGE_WIDTH) {
      int neighbor = nearest_neighbor_field[pixel + direction];
      if (neighbor != -1 && neighbor % IMAGE_WIDTH - direction >= 0 && neighbor % IMAGE_WIDTH - direction < IMAGE_WIDTH) {
	double distance = calcPatchDistance(source_image, target_image, neighbor - direction, pixel, WINDOW_SIZE);
	if (distance < min_distance) {
	  best_nearest_neighbor = neighbor - direction;
	  min_distance = distance;
	}
      } 
  }
    
    if (y + direction >= 0 && y + direction < IMAGE_HEIGHT) {
      int neighbor = nearest_neighbor_field[pixel + direction * IMAGE_WIDTH];
      if (neighbor != -1 && neighbor / IMAGE_WIDTH - direction >= 0 && neighbor / IMAGE_WIDTH - direction < IMAGE_HEIGHT) {
	double distance = calcPatchDistance(source_image, target_image, neighbor - direction * IMAGE_WIDTH, pixel, WINDOW_SIZE);
	if (distance < min_distance) {
	  best_nearest_neighbor = neighbor - direction * IMAGE_WIDTH;
	  min_distance = distance;
	}
      }
    }
    
    int radius = max(IMAGE_WIDTH, IMAGE_HEIGHT);
  int num_attempts = 0;
    while (radius > 0) {
      int x = max(min(current_nearest_neighbor_x + (rand() % (radius * 2 + 1) - radius), IMAGE_WIDTH - 1), 0);
      int y = max(min(current_nearest_neighbor_y + (rand() % (radius * 2 + 1) - radius), IMAGE_HEIGHT - 1), 0);
      int neighbor = y * IMAGE_WIDTH + x;
      double distance = calcPatchDistance(source_image, target_image, neighbor, pixel, WINDOW_SIZE);
      if (distance < min_distance) {
        best_nearest_neighbor = neighbor;
        min_distance = distance;
      }
      radius /= 2;
    }
    if (best_nearest_neighbor != current_nearest_neighbor)
      //cout << pixel % IMAGE_WIDTH << ' ' << pixel / IMAGE_WIDTH << '\t' << best_nearest_neighbor % IMAGE_WIDTH << ' ' << best_nearest_neighbor / IMAGE_WIDTH << '\t' << min_distance << '\t' << current_nearest_neighbor % IMAGE_WIDTH << ' ' << current_nearest_neighbor / IMAGE_WIDTH << '\t' << distance_field[pixel] << endl;
      nearest_neighbor_field[pixel] = best_nearest_neighbor;
    distance_field[pixel] = min_distance;
  }
  
  void calcNearestNeighborField(const Mat &source_image, const Mat &target_image, vector<int> &nearest_neighbor_field, vector<double> &distance_field, const int WINDOW_SIZE)
  {
    const int IMAGE_WIDTH = source_image.cols;
    const int IMAGE_HEIGHT = source_image.rows;
    
    const int NUM_ITERATIONS = 5;
    
    //vector<double> previous_distance_field = distance_field;
    const double DISTANCE_THRESHOLD = 0.000001;
    
    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
      //cout << iteration << endl;
      int direction = 1;
      for (int step = 0; step <= max(IMAGE_WIDTH, IMAGE_HEIGHT) * 2; step++) {
        for (int i = 0; i < max(IMAGE_WIDTH, IMAGE_HEIGHT) * 2; i++) {
          int target_x = IMAGE_WIDTH - 1 - (step - 1 - i);
          int target_y = IMAGE_HEIGHT - 1 - i;
          if (target_x < 0 || target_x >= IMAGE_WIDTH || target_y < 0 || target_y >= IMAGE_HEIGHT)
            continue;
          //cout << target_x << '\t' << target_y << '\t' << IMAGE_WIDTH << '\t' << IMAGE_HEIGHT << '\t' << step << '\t' << i << endl;
          int target_pixel = target_y * IMAGE_WIDTH + target_x;
	  findBetterNearestNeighbor(source_image, target_image, nearest_neighbor_field, distance_field, target_pixel, direction, WINDOW_SIZE);
        }
      }
      
      direction = -1;
      for (int step = 0; step <= max(IMAGE_WIDTH, IMAGE_HEIGHT) * 2; step++) {
        for (int i = 0; i < max(IMAGE_WIDTH, IMAGE_HEIGHT) * 2; i++) {
          int target_x = step - 1 - i;
          int target_y = i;
          //    cout << target_x << '\t' << target_y << endl;
          if (target_x < 0 || target_x >= IMAGE_WIDTH || target_y < 0 || target_y >= IMAGE_HEIGHT)
            continue;
          int target_pixel = target_y * IMAGE_WIDTH + target_x;
	  findBetterNearestNeighbor(source_image, target_image, nearest_neighbor_field, distance_field, target_pixel, direction, WINDOW_SIZE);
        }
      }
    }
  }
}

void findNearestNeighbors(const Mat &source_image, const Mat &target_image, vector<int> &nearest_neighbor_field, const int WINDOW_SIZE)
{
  const int IMAGE_WIDTH = source_image.cols;
  const int IMAGE_HEIGHT = source_image.rows;

  vector<int> source_pixels;
  vector<int> target_pixels;
  
  
  nearest_neighbor_field.assign(IMAGE_WIDTH * IMAGE_HEIGHT, -1);
  vector<double> distance_field(IMAGE_WIDTH * IMAGE_HEIGHT, 1);
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
    nearest_neighbor_field[pixel] = pixel;
    distance_field[pixel] = calcPatchDistance(source_image, target_image, pixel, pixel, WINDOW_SIZE);
  } 
  calcNearestNeighborField(source_image, target_image, nearest_neighbor_field, distance_field, WINDOW_SIZE);
  // for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
  //   cout << pixel << '\t' << nearest_neighbor_field[pixel] << endl;
}
