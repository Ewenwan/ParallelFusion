//
// Created by yanhang on 3/7/16.
//
#include <gflags/gflags.h>
#include "simplestereo.h"

using namespace std;
using namespace cv;
using namespace stereo_base;
using namespace simple_stereo;
DEFINE_int32(resolution, 256, "resolution");
DEFINE_int32(testFrame, 0, "reference frame");
DEFINE_double(weight_smooth, 0.01, "smoothness weight");
DEFINE_int32(downsample, 2, "downsample");
DEFINE_int32(num_threads, 4, "number of threads");

int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: ./SimpleStereo <path-to-data>" << endl;
        return 1;
    }

    google::InitGoogleLogging(argv[1]);
    google::ParseCommandLineFlags(&argc, &argv, true);;

    FileIO file_io(argv[1]);
    SimpleStereo stereo(file_io, FLAGS_testFrame, FLAGS_resolution, FLAGS_downsample, FLAGS_weight_smooth, FLAGS_num_threads);
    stereo.runStereo();
    return 0;
}