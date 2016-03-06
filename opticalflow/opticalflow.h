//
// Created by yanhang on 1/8/16.
//

#ifndef QUADTRACKING_OPTICALFLOW_H
#define QUADTRACKING_OPTICALFLOW_H
#include "OpticalFlow/OpticalFlow.h"
#include "OpticalFlow/GaussianPyramid.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <string>
#include <list>
#include <glog/logging.h>
#ifdef USE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaoptflow.hpp>
#endif

namespace opticalflow_hang {
	namespace interpolate_util{
		template<typename T, int N>
		Eigen::Matrix<double, N, 1> bilinear(const T *const data, const int w, const int h, const Eigen::Vector2d &loc) {
			using namespace Eigen;
			const double epsilon = 0.00001;
			int xl = floor(loc[0] - epsilon), xh = (int) round(loc[0] + 0.5 - epsilon);
			int yl = floor(loc[1] - epsilon), yh = (int) round(loc[1] + 0.5 - epsilon);

			if (loc[0] <= epsilon)
				xl = 0;
			if (loc[1] <= epsilon)
				yl = 0;

			const int l1 = yl * w + xl;
			const int l2 = yh * w + xh;
			if (l1 == l2) {
				Matrix<double, N, 1> res;
				for (size_t i = 0; i < N; ++i)
					res[i] = data[l1 * N + i];
				return res;
			}

			CHECK(!(l1 < 0 || l2 < 0 || l1 >= w * h || l2 >= w * h)) << loc[0] << ' ' << loc[1] << ' '<< w << ' '<< h;

			double lm = loc[0] - (double) xl, rm = (double) xh - loc[0];
			double tm = loc[1] - (double) yl, bm = (double) yh - loc[1];
			Vector4i ind(xl + yl * w, xh + yl * w, xh + yh * w, xl + yh * w);

			std::vector<Matrix<double, N, 1> > v(4);
			for (size_t i = 0; i < 4; ++i) {
				for (size_t j = 0; j < N; ++j)
					v[i][j] = data[ind[i] * N + j];
			}
			if (std::abs(lm) <= epsilon && std::abs(rm) <= epsilon)
				return (v[0] * bm + v[2] * tm) / (bm + tm);

			if (std::abs(bm) <= epsilon && std::abs(tm) <= epsilon)
				return (v[0] * rm + v[2] * lm) / (lm + rm);

			Vector4d vw(rm * bm, lm * bm, lm * tm, rm * tm);
			double sum = vw.sum();
			CHECK_GT(sum, 0);
			return (v[0] * vw[0] + v[1] * vw[1] + v[2] * vw[2] + v[3] * vw[3]) / sum;
		};
	}

	struct FlowFrame{
		FlowFrame(){}
		FlowFrame(const cv::Mat& img_){
			init(img_);
		}
		FlowFrame(const int w, const int h){
			allocate(w,h);
		}
		FlowFrame(const DImage& img_){
			init(img_);
		}
		FlowFrame(const std::string& path){
			readFlowFile(path);
		}

		void init(const cv::Mat& img_);
		void init(const DImage& img_);
		inline int width() const {return w;}
		inline int height() const {return h;}
		inline Eigen::Vector2d getFlowAt(const Eigen::Vector2d& loc)const{
			return interpolation_util::bilinear<double, 2>(img.data(), width(),height(), loc);
		}
		inline void allocate(const int w_, const int h_){
			img.resize(w * h * 2);
			w = w_;
			h = h_;
		}
		inline bool empty() const{
			return img.empty();
		}
		inline void clear(){
			img.clear();
		}
		void setValue(const int x, const int y, const int c, const double v);
		void setValue(const int x, const int y, const Eigen::Vector2d& v);
		inline std::vector<double>& data() {return img;}
		inline const std::vector<double>& data() const {return img;}


		bool readFlowFile(const std::string& path);
		void saveFlowFile(const std::string& path) const;
		inline bool isInsideFlowImage(const Eigen::Vector2d& loc) const{
			return loc[0] > 0 && loc[1] > 0 && loc[0] < w -1 && loc[1] < h - 1;
		}

	private:
		std::vector<double> img;
		int w, h;
	};

	class FlowEstimator{
	public:
		virtual void estimate(const cv::Mat& img1, const cv::Mat& img2, FlowFrame& flow, const int downsample) = 0;
		virtual void downSample(const cv::Mat& input, cv::Mat& output, const int nLevel);
	};

	class FlowEstimatorCPU: public FlowEstimator{
	public:
		virtual void estimate(const cv::Mat& img1, const cv::Mat& img2, FlowFrame& flow, const int nLevel);
	};

#ifdef USE_CUDA
	class FlowEstimatorGPU: public FlowEstimator{
	public:
		FlowEstimatorGPU(){
			brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
		}
		virtual void estimate(const cv::Mat& img1, const cv::Mat& img2, FlowFrame& flow, const int nLevel);
	private:
		cv::Ptr<cv::cuda::BroxOpticalFlow> brox;
	};
#endif

	namespace flow_util {
		void interpolateFlow(const FlowFrame &input, FlowFrame &output, const std::vector<bool> &mask, const bool fillHole = false);
		void warpImage(const cv::Mat &input, cv::Mat &output, const FlowFrame& flow);
		void visualizeFlow(const FlowFrame&, cv::Mat&);
		bool trackPoint(const Eigen::Vector2d& loc, const std::vector<FlowFrame>& flow, const int src, const int tgt, Eigen::Vector2d& res);
		void verifyFlow(const std::vector<FlowFrame>& flow_forward,
						const std::vector<FlowFrame>& flow_backward,
						const std::vector<cv::Mat>& frames,
						std::list<cv::Mat>& verifyimg,
						const int fid, Eigen::Vector2d loc);
		void resizeFlow(const FlowFrame& input, FlowFrame& output, const double ratio, const bool rescale = true);
		void resizeFlow(const FlowFrame& input, FlowFrame& output, const Eigen::Vector2i& dsize, const bool rescale = true);
	}

}//namespace dynamic_rendering

#endif //QUADTRACKING_OPTICALFLOW_H
