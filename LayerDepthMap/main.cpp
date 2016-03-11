#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/photo/photo.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

//#include "SyntheticSceneGenerator.h"
#include "LayerDepthRepresenter.h"
#include "utils.h"
#include "TRW_S/MRFEnergy.h"
//#include "SRMP.h"
//#include "GeometryCalculation.h"
//#include "SurfaceMaskCalculator.h"

//#include "ImageSegmenter.h"
//#include "PatchMatcher.h"


using namespace std;
using namespace cv;

// DEFINE_double(opposite_distance_threshold, 0.01, "The distance threshold determining whether a point is on one side of a surface or not");
// DEFINE_double(num_opposite_outliers_threshold_ratio, 0.01, "The ratio threshold determining whether a surface is one on side of a surface or not");

DEFINE_int32(scene_index, 10000, "Scene index.");
DEFINE_int32(dataset_index, 1, "dataset index.");
DEFINE_int32(num_layers, 4, "The number of layers.");
DEFINE_string(scene_name, "cse013", "Scene name.");
DEFINE_int32(scene_rotation_angle, 0, "The rotation angle of the scene.");

DEFINE_int32(num_threads, 2, "The number of threads.");
DEFINE_int32(num_iterations, 30, "The number of iterations.");

DEFINE_int32(num_proposals_in_total, 1, "The number of proposals in total.");
DEFINE_int32(num_proposals_from_others, 0, "The number of proposals from others.");
DEFINE_int32(solution_exchange_interval, 3, "The number of iterations between consecutive solution exchanges.");
DEFINE_int32(result_index, 0, "The index of the result.");
DEFINE_bool(use_monitor_thread, false, "Whether monitor object is used.");


int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_log_dir = "Log";
  //  FLAGS_logtostderr = false;
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << FLAGS_scene_name;

  {
    time_t timer;
    time(&timer);
    struct tm today = {0};
    today.tm_hour = today.tm_min = today.tm_sec = 0;
    today.tm_mon = 2;
    today.tm_mday = 9;
    today.tm_year = 116;
    LOG(INFO) << difftime(timer, mktime(&today)) << '\t' << -1 << '\t' << -1 << '\t' << 0;
  }


  srand(0);
  
  vector<double> point_cloud;
  Mat image;
  Mat ori_image;
  vector<double> ori_point_cloud;
  MatrixXd projection_matrix;
  char scene_type = 'R';
  bool first_time = false;
  // if (argc <= 2)
  //   first_time = true;
  // else {
  //   string first_time_str(argv[2]);
  //   if (first_time_str.compare("true") == 0)
  //     first_time = true;
  // }
  
  stringstream result_dir_ss;
  result_dir_ss << "Results/scene_" << FLAGS_scene_index;
  
  struct stat sb;
  if (stat(result_dir_ss.str().c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
    mkdir(result_dir_ss.str().c_str(), 0777);
  stringstream cache_dir_ss;
  cache_dir_ss << "Cache/scene_" << FLAGS_scene_index;
  if (stat(cache_dir_ss.str().c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
    mkdir(cache_dir_ss.str().c_str(), 0777);
  
  if (FLAGS_dataset_index == 0) {
    if (scene_type == 'R') {
      char *image_filename = new char[100];
      sprintf(image_filename, "Inputs/image_%02d.bmp", FLAGS_scene_index);
      image = imread(image_filename);
      ori_image = image.clone();
      // Size ori_image_size(image.cols / 4 * 4, image.rows / 4 * 4);
      // ori_image = Mat(ori_image_size, CV_8UC3);
      // resize(image, ori_image, ori_image_size);
    
      char *point_cloud_filename = new char[100];
      sprintf(point_cloud_filename, "Inputs/point_cloud_%02d.txt", FLAGS_scene_index);
      point_cloud = loadPointCloud(point_cloud_filename);
      ori_point_cloud = point_cloud;
      //point_cloud = normalizePointCloudByZ(point_cloud);
    
      // bool segment_image = false;
      // if (segment_image) {
      //   // Mat disp_image = drawDispImage(point_cloud, image.cols, projection_matrix);
      //   // normalize(disp_image, disp_image, 0, 128, NORM_MINMAX);
      //   // imwrite("Results/disp_image.bmp", disp_image);

      //   //point_cloud = inpaintPointCloud(point_cloud, image.cols, image.rows);
      //   double focal_length = 520;
      //   //estimateCameraParameters(point_cloud, image.cols, image.rows, focal_length);
      //   ImageSegmenter image_segmenter(image, point_cloud, focal_length, scene_index);
      //   vector<int> segmentation = image_segmenter.getSegmentation();
      //   Mat segmentation_image = drawSegmentationImage(segmentation, image.cols);
      //   imwrite("Results/segmentation_image.bmp", segmentation_image);
      //   exit(1);
      // }
    
      // for (int y = 0; y < ori_image.rows; y++)
      //   for (int x = 0; x < ori_image.cols; x++)
      // 	if (x < ori_image.cols / 2)
      // 	  ori_image.at<Vec3b>(y, x) = Vec3b(0, 255, 0);
      // 	else
      // 	  ori_image.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
    
      // char *segmentation_filename = new char[100];
      // sprintf(segmentation_filename, "Inputs/final_segmentation_%02d.txt", scene_index);
      // segmentation = loadSegmentation(segmentation_filename);
      // segmentation = deleteSmallSegments(segmentation, image.cols, 3);
      //segmentation = vector<int>(ori_image.cols * ori_image.rows);

      if (first_time) {
	// double focal_length, cx, cy;
	// estimateCameraParameters(point_cloud, image.cols, image.rows, focal_length, cx, cy);
	//cout << focal_length << '\t' << cx << '\t' << cy << endl;
      
	vector<double> inpainted_point_cloud = point_cloud; //inpaintPointCloud(point_cloud, image.cols, image.rows);
	point_cloud = inpainted_point_cloud;
      
	//point_cloud = smoothPointCloud(point_cloud, segmentation, image.cols, image.rows);
	if (true) {
	  stringstream depth_values_filename;
	  depth_values_filename << "Results/scene_" << FLAGS_scene_index << "/" << "depth_values_ori";
	  // depth_values_filename.fill('0');
	  // depth_values_filename.width(3);
	  // depth_values_filename << scene_index;
	  ofstream depth_values_out_str(depth_values_filename.str().c_str());
	  depth_values_out_str << image.cols << '\t' << image.rows << endl;
	  for (int point_index = 0; point_index < inpainted_point_cloud.size() / 3; point_index++)
	    depth_values_out_str << inpainted_point_cloud[point_index * 3 + 2] << endl;
	  depth_values_out_str.close();
	} else {
	  // vector<double> triangle_vertices = triangulateSceneUsingPointCloud(inpainted_point_cloud, image.cols, image.rows, focal_length, cx, cy);
	  // stringstream triangle_vertices_filename;
	  // triangle_vertices_filename << "Results/scene_" << scene_index << "/" << "triangles_ori_";
	  // triangle_vertices_filename.fill('0');
	  // triangle_vertices_filename.width(3);
	  // triangle_vertices_filename << scene_index;
	  // ofstream triangle_vertices_out_str(triangle_vertices_filename.str().c_str());
	  // triangle_vertices_out_str << triangle_vertices.size() / 9 << endl;
	  // for (int value_index = 0; value_index < triangle_vertices.size(); value_index++)
	  //   triangle_vertices_out_str << triangle_vertices[value_index] << endl;
	  // triangle_vertices_out_str.close();
	}
      }    

      //    cropRegion(image, point_cloud, segmentation, 100, 275, 110, 55);
    
      zoomScene(image, point_cloud, 0.35, 0.35);
      //      zoomScene(ori_image, ori_point_cloud, 1.0 * (image.cols * static_cast<int>(ori_image.cols / image.cols)) / ori_image.cols, 1.0 * (image.cols * static_cast<int>(ori_image.cols / image.cols)) / ori_image.cols);
      
      stringstream point_cloud_ply_filename;
      point_cloud_ply_filename << "Results/scene_" << FLAGS_scene_index << "/point_cloud.ply";
      savePointCloudAsPly(point_cloud, point_cloud_ply_filename.str().c_str());

      stringstream point_cloud_mesh_filename;
      point_cloud_mesh_filename << "Results/scene_" << FLAGS_scene_index << "/point_cloud_mesh.ply";
      savePointCloudAsMesh(point_cloud, point_cloud_mesh_filename.str().c_str());

      projection_matrix = MatrixXd::Zero(3, 4);
      vector<double> camera_parameters = calcCameraParameters(point_cloud, image.cols);
      projection_matrix(0, 0) = camera_parameters[0];
      projection_matrix(0, 2) = camera_parameters[1];
      projection_matrix(1, 1) = camera_parameters[2];
      projection_matrix(1, 2) = camera_parameters[3];
      projection_matrix(2, 2) = 1;
    } else if (scene_type == 'S') {
      // SyntheticSceneGenerator generator;
      // generator.generateScene(scene_index);
      // projection_matrix = generator.getProjectionMatrix();
      // char *point_cloud_filename = new char[100];
      // sprintf(point_cloud_filename, "GeneratedScene/point_cloud_%d", scene_index);
      // point_cloud = loadPointCloud(point_cloud_filename);
      // point_cloud = normalizePointCloudByZ(point_cloud);
      // char *image_filename = new char[100];
      // sprintf(image_filename, "GeneratedScene/image_%d.bmp", scene_index);
      // image = imread(image_filename, 0);
      // ori_image = image.clone();
      // char *segmentation_filename = new char[100];
      // sprintf(segmentation_filename, "GeneratedScene/segmentation_%d", scene_index);
      // segmentation = loadSegmentation(segmentation_filename);
      // imwrite("Test/segmentation_image.bmp", drawSegmentationImage(segmentation, image.cols));
      // scene_index += 10000;
      // 							      //      zoomScene(image, point_cloud, 5, 5);
      // stringstream disp_image_filename;
      // disp_image_filename << "Test/disp_image.bmp";
      // imwrite(disp_image_filename.str(), drawDispImage(point_cloud, image.cols, image.rows)); //projection_matrix));
      exit(1);
    }
  
    stringstream zoomed_image_filename;
    zoomed_image_filename << "Results/scene_" << FLAGS_scene_index << "/zoomed_image.bmp";
    imwrite(zoomed_image_filename.str(), image);

    // bool test_image_completion = false;
    // if (test_image_completion) {
    //   Mat test_image = image.clone();
    //   // Mat test_image = Mat::zeros(100, 100, CV_8UC3);
    //   // for (int x = 0; x < test_image.cols; x++)
    //   //   test_image.at<Vec3b>(test_image.rows / 2, x) = Vec3b(255, 255, 255);
    //   PatchMatcher patch_matcher(test_image);
    //   vector<int> segment_pixels, unknown_pixels;
    //   for (int y = test_image.rows / 8 * 3; y < test_image.rows / 8 * 5; y++) {
    // 	for (int x = test_image.cols / 8 * 3; x < test_image.cols / 8 * 6; x++) {
    // 	  int pixel = y * test_image.cols + x;
    // 	  segment_pixels.push_back(pixel);
    // 	  if (x < test_image.cols / 2 + 10)
    // 	    unknown_pixels.push_back(pixel);
    // 	}
    //   }
    //   patch_matcher.matchPatch(segment_pixels, unknown_pixels);
    //   Mat image_completed = patch_matcher.getCompletedImage();
    //   imwrite("Test/image_completed.bmp", image_completed);
    //   exit(1);
    // }

  } else {
    
    // char *image_filename = new char[100];
    // sprintf(image_filename, "Inputs/image_%02d.bmp", scene_index);
    
    //    image = imread("Inputs/cse013_0h.png");
    stringstream image_filename;
    image_filename << "Inputs/" << FLAGS_scene_name << "_" << FLAGS_scene_rotation_angle << ".png";
    image = imread(image_filename.str());
    ori_image = image.clone();

    stringstream point_cloud_filename;
    point_cloud_filename << "Inputs/" << FLAGS_scene_name << "_" << FLAGS_scene_rotation_angle << ".obj";
    point_cloud = readPointCloudFromObj(point_cloud_filename.str(), image.cols, image.rows, FLAGS_scene_rotation_angle);
    //point_cloud = rotatePointCloud(point_cloud, rotation_angle);
    //point_cloud = readPointCloudFromObj("Inputs/cse013_0h.obj", image.cols, image.rows);
    ori_point_cloud = point_cloud;

    //    zoomScene(image, point_cloud, 0.2, 0.2);
    zoomScene(image, point_cloud, 1.0 * 200 / image.cols, 1.0 * 200 / image.cols);
    zoomScene(ori_image, ori_point_cloud, 1.0 * (image.cols * static_cast<int>(ori_image.cols / image.cols)) / ori_image.cols, 1.0 * (image.cols * static_cast<int>(ori_image.cols / image.cols)) / ori_image.cols);

    // int new_index = 66 * image.cols + 111;
    // cout << point_cloud[new_index * 3 + 0] << '\t' << point_cloud[new_index * 3 + 1] << '\t' << point_cloud[new_index * 3 + 2] << endl;
    // int ori_index = 369 * ori_image.cols + 533;
    // cout << ori_point_cloud[ori_index * 3 + 0] << '\t' << ori_point_cloud[ori_index * 3 + 1] << '\t' << ori_point_cloud[ori_index * 3 + 2] << endl;
    // exit(1);
    
    // cout << point_cloud[(116 * image.cols + 57) * 3 + 0] << '\t' << point_cloud[(116 * image.cols + 57) * 3 + 1] << '\t' << point_cloud[(116 * image.cols + 57) * 3 + 2] << endl;
    // exit(1);
    // double max_depth = 0;
    // for (int i = 0; i < point_cloud.size() / 3; i++)
    //   if (point_cloud[i * 3 + 2] > max_depth)
    // 	max_depth = point_cloud[i * 3 + 2];
    // cout << max_depth << endl;
    // exit(1);
    
    stringstream zoomed_image_filename;
    zoomed_image_filename << "Results/scene_" << FLAGS_scene_index << "/zoomed_image.bmp";
    imwrite(zoomed_image_filename.str(), image);

    stringstream large_image_filename;
    large_image_filename << "Results/scene_" << FLAGS_scene_index << "/large_image.bmp";
    imwrite(large_image_filename.str(), ori_image);
  }

  bool check_energy_diff = false;
  if (check_energy_diff) {
    Mat test_image = Mat::zeros(image.rows, image.cols, CV_8UC1);
    ifstream energy_in_str_1("Test/energy_0");
    ifstream energy_in_str_2("Test/energy_1");
    for (int y = 0; y < test_image.rows; y++) {
      for (int x = 0; x < test_image.cols; x++) {
        int pixel = y * test_image.cols + x;
        int energy_1;
        energy_in_str_1 >> energy_1;
        int energy_2;
        energy_in_str_2 >> energy_2;
        if (energy_1 != energy_2)
          cout << pixel % image.cols << '\t' << pixel / image.cols << '\t' << energy_1 << '\t' << energy_2 << endl;
        test_image.at<uchar>(y, x) = min(max(1.0 * (energy_1 - energy_2) / 100 * 128 + 128, 0.0), 255.0);
      }
    }
    imwrite("Test/energy_diff_image.bmp", test_image);
    exit(1);
  }

  bool check_blending = false;
  if (check_blending) {
    Mat texture_image = imread("Results/scene_10001/texture_image_3.bmp");
    Mat mask = Mat::ones(texture_image.rows, texture_image.cols, texture_image.depth()) * 255;
    // for (int y = 0; y < mask.rows / 2; y++)
    //   for (int x = 0; x < mask.cols; x++)
    // 	mask.at<Vec3b>(y, x) = Vec3b(0, 0, 0);

    Mat result;
    //seamlessClone(texture_image, texture_image, mask, Point(texture_image.cols / 2, texture_image.rows / 2), result, MIXED_CLONE);
    GaussianBlur(texture_image, result, Size(5, 5), 0, 0);
    GaussianBlur(result, result, Size(5, 5), 0, 0);
    GaussianBlur(result, result, Size(5, 5), 0, 0);
    imshow("texture_image", result);
    waitKey();
    exit(1);
  }

  if (true) {
    stringstream texture_ori_filename;
    texture_ori_filename << "Results/scene_" << FLAGS_scene_index << "/" << "texture_image_ori.bmp";
    if (imread(texture_ori_filename.str()).empty())
      imwrite(texture_ori_filename.str(), ori_image);

    
    stringstream depth_values_filename;
    depth_values_filename << "Results/scene_" << FLAGS_scene_index << "/" << "depth_values_ori";
    if (!ifstream(depth_values_filename.str().c_str())) {
      vector<double> inpainted_point_cloud = point_cloud; //inpaintPointCloud(ori_point_cloud, ori_image.cols, ori_image.rows);
      ofstream depth_values_out_str(depth_values_filename.str().c_str());
      depth_values_out_str << ori_image.cols << '\t' << ori_image.rows << endl;
      for (int point_index = 0; point_index < inpainted_point_cloud.size() / 3; point_index++)
	depth_values_out_str << inpainted_point_cloud[point_index * 3 + 2] << endl;
      depth_values_out_str.close();
    }
  
    stringstream point_cloud_ply_filename;
    point_cloud_ply_filename << "Results/scene_" << FLAGS_scene_index << "/point_cloud.ply";
    if (!ifstream(point_cloud_ply_filename.str().c_str()))
      savePointCloudAsPly(point_cloud, point_cloud_ply_filename.str().c_str());
  
    stringstream disp_image_filename;
    disp_image_filename << "Results/scene_" << FLAGS_scene_index << "/disp_image.bmp";
    if (imread(disp_image_filename.str()).empty())
      imwrite(disp_image_filename.str(), drawDispImage(point_cloud, image.cols, image.rows));

    stringstream ori_disp_image_filename;
    ori_disp_image_filename << "Results/scene_" << FLAGS_scene_index << "/disp_image_ori.bmp";
    if (imread(ori_disp_image_filename.str()).empty())
      imwrite(ori_disp_image_filename.str(), drawDispImage(ori_point_cloud, ori_image.cols, ori_image.rows));
  }
  cout << "scene: " << FLAGS_scene_index << endl;
  //exit(1);
  
  RepresenterPenalties penalties;
  DataStatistics statistics;
  if (FLAGS_dataset_index == 0 && false) {
    penalties.depth_inconsistency_pen = 2000;
    penalties.normal_inconsistency_pen = 200;
    penalties.color_inconsistency_pen = 10;
    penalties.distance_2D_pen = 0;
    penalties.close_parallel_surface_pen = 0;
    penalties.layer_empty_pen = 0;
  
    penalties.smoothness_pen = 10000;
    penalties.smoothness_small_constant_pen = 1;
    penalties.smoothness_concave_shape_pen = 500;
    penalties.smoothness_segment_splitted_pen = 0;
    penalties.smoothness_spurious_empty_pen = 0;
    penalties.smoothness_boundary_pen = 500;

    penalties.other_viewpoint_depth_change_pen = 2000;
    penalties.other_viewpoint_depth_conflict_pen = 200000;
  
    penalties.surface_pen = 20000;
    penalties.layer_pen = 0;
    penalties.surface_splitted_pen = 0;
  
    penalties.data_cost_depth_change_ratio = 4;
    //penalties.data_cost_angle_diff_ratio = 1;
    //penalties.data_cost_color_likelihood_ratio = 5;
    penalties.data_cost_non_plane_ratio = 0.05;
    penalties.smoothness_empty_non_empty_ratio = 0.05;

    //penalties.large_pen = 10000;
    penalties.huge_pen = 1000000;

    penalties.data_term_layer_decrease_ratio = 1;
    penalties.smoothness_term_layer_decrease_ratio = 1; //sqrt(0.5);
    
    
    statistics.pixel_fitting_distance_threshold = 0.03;
    statistics.pixel_fitting_angle_threshold = 30 * M_PI / 180;
    statistics.pixel_fitting_color_likelihood_threshold = -20;
    statistics.depth_diff_var = 0.01;
    statistics.similar_angle_threshold = 20 * M_PI / 180;
    // statistics.fitting_color_likelihood_threshold = 0.01;
    // statistics.parallel_angle_threshold = 10 * M_PI / 180;
    statistics.viewpoint_movement = 0.1;
    //statistics.color_diff_threshold = 0.5;
    statistics.depth_conflict_threshold = statistics.pixel_fitting_distance_threshold;
    statistics.depth_change_smoothness_threshold = 0.02;
    statistics.bspline_surface_num_pixels_threshold = image.cols * image.rows / 50;
    statistics.background_depth_diff_tolerance = 0.05;
  } else {
    penalties.depth_inconsistency_pen = 2000;
    penalties.normal_inconsistency_pen = 200;
    penalties.color_inconsistency_pen = 10;
    penalties.distance_2D_pen = 0;
    penalties.close_parallel_surface_pen = 0;
    penalties.layer_empty_pen = 0;
  
    penalties.smoothness_pen = 10000;
    penalties.smoothness_small_constant_pen = 1;
    penalties.smoothness_concave_shape_pen = 5000;
    penalties.smoothness_segment_splitted_pen = 0;
    penalties.smoothness_spurious_empty_pen = 0;
    penalties.smoothness_boundary_pen = 500;

    penalties.other_viewpoint_depth_change_pen = 2000;
    penalties.other_viewpoint_depth_conflict_pen = 200000;
  
    penalties.surface_pen = 20000;
    penalties.layer_pen = 0;
    penalties.surface_splitted_pen = 0;
  
    penalties.data_cost_depth_change_ratio = 4;
    //penalties.data_cost_angle_diff_ratio = 1;
    //penalties.data_cost_color_likelihood_ratio = 5;
    penalties.data_cost_non_plane_ratio = 0.05;
    penalties.smoothness_empty_non_empty_ratio = 0.05;

    //penalties.large_pen = 10000;
    penalties.huge_pen = 1000000;

    penalties.data_term_layer_decrease_ratio = 1;
    penalties.smoothness_term_layer_decrease_ratio = 1; //sqrt(0.5);
    
    
    statistics.pixel_fitting_distance_threshold = 0.03;
    statistics.pixel_fitting_angle_threshold = 30 * M_PI / 180;
    statistics.pixel_fitting_color_likelihood_threshold = -20;
    statistics.depth_diff_var = 0.01;
    statistics.similar_angle_threshold = 20 * M_PI / 180;
    // statistics.fitting_color_likelihood_threshold = 0.01;
    // statistics.parallel_angle_threshold = 10 * M_PI / 180;
    statistics.viewpoint_movement = 0.1;
    //statistics.color_diff_threshold = 0.5;
    statistics.depth_conflict_threshold = statistics.pixel_fitting_distance_threshold;
    statistics.depth_change_smoothness_threshold = 0.02;
    statistics.bspline_surface_num_pixels_threshold = image.cols * image.rows / 50;
    statistics.background_depth_diff_tolerance = 0.05;
  }
  
  // if (scene_type == 'S') {
  //   cvtColor(ori_image, ori_image, CV_GRAY2BGR);
  //   //point_cloud = projectPointCloud(point_cloud, projection_matrix);
  //   //scene_index += 10000;
  // }

  // vector<int> pixels;
  // for (int y = 46; y < 76; y++)
  //   for (int x = 98; x < 128; x++)
  //     pixels.push_back(y * image.cols + x);
  // BSplineSurface surface(point_cloud, pixels, image.cols, image.rows, 5, 5, 3);
  // exit(1);
  
  // vector<double> normals = calcNormals(point_cloud, image.cols, image.rows, 21);
  // Mat normal_image = Mat(image.rows, image.cols, CV_8UC3);
  // for (int y = 0; y < image.rows; y++) {
  //   for (int x = 0; x < image.cols; x++) {
  //     int pixel = y * image.cols + x;
  //     vector<double> normal(normals.begin() + pixel * 3, normals.begin() + (pixel + 1) * 3);
  //     Vec3b color;
  //     color[0] = max(min(static_cast<double>((normal[0] + 1) * 128), 255.0), 0.0) + 0.5;
  //     color[1] = max(min(static_cast<double>((normal[1] + 1) * 128), 255.0), 0.0) + 0.5;
  //     color[2] = max(min(static_cast<double>((normal[2] + 1) * 128), 255.0), 0.0) + 0.5;
  //     normal_image.at<Vec3b>(y, x) = color;
  //   }
  // }
  // imwrite("Test/normal_image.bmp", normal_image);
  // cout << "done" << endl;
  // exit(1);


  bool test_image_completion = false;
  if (test_image_completion) {
    Mat test_image = imread("Test/image_for_completion_4.bmp");
    // resize(test_image, test_image, Size(200, 100), 0.4, 0.4, INTER_NEAREST);
    // imwrite("Test/image_for_completion.bmp", test_image);
    // exit(1);
    // vector<int> segment_pixels;
    // vector<int> hole_pixels;
    // for (int pixel = 0; pixel < test_image.cols * test_image.rows; pixel++) {
    //   Vec3b color = test_image.at<Vec3b>(pixel / test_image.cols, pixel % test_image.cols);
    //   if (color != Vec3b(255, 255, 255))
    // 	segment_pixels.push_back(pixel);
    //   if (color == Vec3b(0, 0, 0)) {
    //     hole_pixels.push_back(pixel);
    // 	//	test_image.at<Vec3b>(pixel / test_image.cols, pixel % test_image.cols) = Vec3b(255, 255, 255);
    //   }
    // }
    
    // PatchMatcher patch_matcher(test_image);
    // Mat mask_image;
    // cvtColor(test_image, mask_image, CV_RGB2GRAY);
    // for (int pixel = 0; pixel < test_image.cols * test_image.rows; pixel++) {
    //   Vec3b color = test_image.at<Vec3b>(pixel / test_image.cols, pixel % test_image.cols);
    //   if (color != Vec3b(255, 255, 255) && color != Vec3b(0, 0, 0))
    // 	mask_image.at<uchar>(pixel / test_image.cols, pixel % test_image.cols) = 128;
    // }
    
    // patch_matcher.matchPatch(mask_image);
    // Mat image_completed = patch_matcher.getCompletedImage();
    // imwrite("Test/image_completed.bmp", image_completed);
    // exit(1);
  }

  PipelineParams pipeline_params;
  pipeline_params.num_threads = FLAGS_num_threads;
  pipeline_params.num_iterations = FLAGS_num_iterations;
  pipeline_params.num_proposals_in_total = FLAGS_num_proposals_in_total;
  pipeline_params.num_proposals_from_others = FLAGS_num_proposals_from_others;
  pipeline_params.solution_exchange_interval = FLAGS_solution_exchange_interval;
  pipeline_params.use_monitor_thread = FLAGS_use_monitor_thread;
  
  LayerDepthRepresenter representer(image, point_cloud, penalties, statistics, FLAGS_scene_index, ori_image, ori_point_cloud, first_time, FLAGS_num_layers, pipeline_params);

  // A[1][1] = 2;
  // A[1][2] = 1;
  // vector<double> blc(3, -MSK_INFINITY);
  // vector<double> buc(3);
  // buc[0] = 4;
  // buc[1] = 12;
  // buc[2] = 1;
  // vector<MSKboundkeye> bkc(3, MSK_BK_UP);
  
  // vector<double> blx(2, 0);
  // vector<double> bux(2, MSK_INFINITY);
  // vector<MSKboundkeye> bkx(2, MSK_BK_LO);
  
  // LinearProgrammingSolver solver = LinearProgrammingSolver(2, 3, c, A, blc, buc, bkc, blx, bux, bkx);
  // solver.solve();
  // return 0;
}
