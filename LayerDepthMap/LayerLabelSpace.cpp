#include "LayerLabelSpace.h"


using namespace std;


void LayerLabelSpace::appendSpace(const LayerLabelSpace & rhs)
//void LayerLabelSpace::appendSpace(const shared_ptr<LabelSpace<int> > rhs)
{
  unionSpace(rhs);
}

void LayerLabelSpace::unionSpace(const LayerLabelSpace &rhs)
//void LayerLabelSpace::unionSpace(const shared_ptr<LabelSpace<int> > rhs)
{
  if(num_nodes_ == 0) {
    init(rhs.getNumNode());
    NUM_PIXELS_ = rhs.getNumPixels();
    NUM_LAYERS_ = rhs.getNumLayers();
    segments_.clear();
  }      
  
  map<int, Segment> rhs_segments = rhs.getSegments();
  
  int new_segment_id = 0;
  map<int, int> unique_segment_id_map;
  for (map<int, Segment>::const_iterator segment_it = segments_.begin(); segment_it != segments_.end(); segment_it++) {
    if (unique_segment_id_map.count(segment_it->second.getSegmentId()) == 0)
      unique_segment_id_map[segment_it->second.getSegmentId()] = new_segment_id++;
    //cout << segment_it ->first << '\t' << segment_it->second.getSegmentId() << endl;
  }
  for (map<int, Segment>::const_iterator rhs_segment_it = rhs_segments.begin(); rhs_segment_it != rhs_segments.end(); rhs_segment_it++)
    if (unique_segment_id_map.count(rhs_segment_it->second.getSegmentId()) == 0)
      unique_segment_id_map[rhs_segment_it->second.getSegmentId()] = new_segment_id++;

  
  const int UNION_NUM_SEGMENTS = unique_segment_id_map.size();
  
  map<int, Segment> union_segments;
  vector<vector<int> > union_labels(NUM_PIXELS_);
  
  if (getNumSegments() > 0) {
    const int NUM_SEGMENTS = getNumSegments();
    for (map<int, Segment>::const_iterator segment_it = segments_.begin(); segment_it != segments_.end(); segment_it++) {
      int union_segment_id = unique_segment_id_map[segment_it->second.getSegmentId()];
      if (union_segments.count(union_segment_id) == 0)
        union_segments[union_segment_id] = segment_it->second;
    }
  
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      const vector<int> labels = getLabelOfNode(pixel);
      for (vector<int>::const_iterator label_it = labels.begin(); label_it != labels.end(); label_it++) {
	int new_label = 0;
	for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	  int segment_id = *label_it / static_cast<int>(pow(NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index)) % (NUM_SEGMENTS + 1);
	  // if (getNumSegments() > 0 && rhs.getNumSegments() > 0)
          //   cout << pixel << '\t' << *label_it << '\t' << segment_id << '\t' << endl;
      
          if (segment_id < NUM_SEGMENTS)
	    new_label += unique_segment_id_map[segments_.at(segment_id).getSegmentId()] * pow(UNION_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index);
          else
	    new_label += UNION_NUM_SEGMENTS * pow(UNION_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index);
        }
        union_labels[pixel].push_back(new_label);
      }
    }  
  }

  if (rhs.getNumSegments() > 0) {
    const int RHS_NUM_SEGMENTS = rhs.getNumSegments();
    for (map<int, Segment>::const_iterator segment_it = rhs_segments.begin(); segment_it != rhs_segments.end(); segment_it++) {
      int union_segment_id = unique_segment_id_map[segment_it->second.getSegmentId()];
      if (union_segments.count(union_segment_id) == 0)
        union_segments[union_segment_id] = segment_it->second;
    }
    
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      const vector<int> labels = rhs.getLabelOfNode(pixel);
      for (vector<int>::const_iterator label_it = labels.begin(); label_it != labels.end(); label_it++) {
	int new_label = 0;
	for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	  int segment_id = *label_it / static_cast<int>(pow(RHS_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index)) % (RHS_NUM_SEGMENTS + 1);
	  if (segment_id < RHS_NUM_SEGMENTS)
	    new_label += unique_segment_id_map[rhs_segments.at(segment_id).getSegmentId()] * pow(UNION_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index);
	  else
	    new_label += UNION_NUM_SEGMENTS * pow(UNION_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index);
        }
	if (find(union_labels[pixel].begin(), union_labels[pixel].end(), new_label) == union_labels[pixel].end())
	  union_labels[pixel].push_back(new_label);
      }
    }
  }

  //cout << "num: " << getNumSegments() << '\t' << rhs.getNumSegments() << '\t' << UNION_NUM_SEGMENTS << endl;
  
  if (UNION_NUM_SEGMENTS == 0)
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      if (union_labels[pixel].size() == 0)
        union_labels[pixel].push_back(0);

  CHECK_EQ(union_segments.size(), UNION_NUM_SEGMENTS) << "The number of segments in union is not consistent" << endl;
  
  // if (getNumSegments() > 0 && rhs.getNumSegments() > 0)
  //   exit(1);

  label_space_ = union_labels;
  segments_ = union_segments;
  addIndicatorLabels();
  num_nodes_ = NUM_PIXELS_ + segments_.size() * NUM_LAYERS_;
}

void LayerLabelSpace::addIndicatorLabels()
{
  vector<int> indicator_labels(2);
  indicator_labels[0] = 0;
  indicator_labels[1] = 1;
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    for (int segment_index = 0; segment_index < segments_.size(); segment_index++)
      label_space_.push_back(indicator_labels);
}
