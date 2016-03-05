//
// Created by yanhang on 3/4/16.
//

#include <iostream>
#include <gflags/gflags.h>

#include "scestereo.h"

using namespace std;
using namespace sce_stereo;

DEFINE_int32(testFrame, -1, "test frame");
DEFINE_int32(tWindow, -1, "tWindow");
DEFINE_int32(resolution, 64, "resolution");

int main(int argc, char **argv){
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    CHECK_GE(FLAGS_testFrame, 0);
    CHECK_GE(FLAGS_tWindow, 2);
    if(argc < 2){
        cerr << "Not enough parameter" << endl;
        return 1;
    }

    FileIO file_io(argv[1]);

    SceStereo stereo(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_resolution);
    stereo.runStereo();

    return 0;
}

