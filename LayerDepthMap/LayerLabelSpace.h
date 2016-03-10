#ifndef LAYER_LABEL_SPACE_H__
#define LAYER_LABEL_SPACE_H__

#include <vector>
#include <glog/logging.h>
#include <algorithm>
#include <map>

#include "Segment.h"

#include "../base/LabelSpace.h"

//Label space management
class LayerLabelSpace : public ParallelFusion::LabelSpace<int> {

 public:
  void appendSpace(const LayerLabelSpace &rhs);
  void unionSpace(const LayerLabelSpace &rhs);

  //  void appendSpace(const std::shared_ptr<ParallelFusion::LabelSpace<int> > rhs);
  //void unionSpace(const std::shared_ptr<ParallelFusion::LabelSpace<int> > rhs);

  std::map<int, Segment> getSegments() const { return segments_; };
  int getNumPixels() const { return NUM_PIXELS_; };
  int getNumLayers() const { return NUM_LAYERS_; };
  int getNumSegments() const { return segments_.size(); };
  
  void setSegments(const std::map<int, Segment> &segments) { segments_ = segments; };
  void setNumPixels(const int NUM_PIXELS) { NUM_PIXELS_ = NUM_PIXELS; };
  void setNumLayers(const int NUM_LAYERS) { NUM_LAYERS_ = NUM_LAYERS; };
  
 private:

  std::map<int, Segment> segments_;
  int NUM_PIXELS_;
  int NUM_LAYERS_;
  void addIndicatorLabels();
  
};
  
#endif
