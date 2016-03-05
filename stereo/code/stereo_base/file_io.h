#ifndef FILE_IO_H
#define FILE_IO_H

#include <string>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <assert.h>

namespace dynamic_stereo{
    class FileIO {
    public:
        void init() {
            int curid = startid;
            while (true) {
                char buffer[1024] = {};
                sprintf(buffer, "%s/images/%s%05d.jpg", directory.c_str(), imagePrefix.c_str(), curid);
                std::ifstream fin(buffer);
                if (!fin.is_open()) {
                    fin.close();
                    break;
                }
                curid++;
                fin.close();
            }
            framenum = curid;
        }

        FileIO(std::string directory_) : imagePrefix("image"), startid(0), directory(directory_) {
            //get the number of frame
            init();
        }

        FileIO(const std::string &directory_, const std::string &imagePrefix_, const int startid_) :
                directory(directory_), imagePrefix(imagePrefix_), startid(startid_) {
            init();
        }


        inline int getTotalNum() const { return framenum; }

        inline std::string getImageDirectory() const{
            return getDirectory() + "/images/";
        }
        inline std::string getImage(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/images/%s%05d.jpg", directory.c_str(), imagePrefix.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getPose(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/pose%05d.txt", getDirectory().c_str(), id);
            return std::string(buffer);
        }


        inline std::string getPoseDirectory() const{
            return getDirectory() + "/pose/";
        }

        inline std::string getSfMDirectory() const{
            return getDirectory() + "/sfm/";
        }
        inline std::string getMvgDirectory() const{
            return getDirectory() + "/mvg/";
        }
        inline std::string getReconstruction() const{
            return getSfMDirectory() + "reconstruction.recon";
        }

        inline std::string getDirectory() const {
            return directory;
        }
    private:
        const std::string imagePrefix;
        const int startid;
        const std::string directory;
        int framenum;
    };

}

#endif

