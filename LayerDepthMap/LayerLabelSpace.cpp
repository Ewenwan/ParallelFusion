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

  set<int> unique_segment_ids;
  {
    const int NUM_SEGMENTS = getNumSegments();
    if (NUM_SEGMENTS > 0) {
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	const vector<int> labels = getLabelOfNode(pixel);
	for (vector<int>::const_iterator label_it = labels.begin(); label_it != labels.end(); label_it++) {
	  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	    int segment_id = *label_it / static_cast<int>(pow(NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index)) % (NUM_SEGMENTS + 1);
	    if (segment_id == NUM_SEGMENTS)
	      continue;
	    unique_segment_ids.insert(segment_id);
	  }
	}
	if (unique_segment_ids.size() == NUM_SEGMENTS)
	  break;
      }
    }
  }
  set<int> rhs_unique_segment_ids;
  {
    const int RHS_NUM_SEGMENTS = rhs.getNumSegments();
    if (RHS_NUM_SEGMENTS > 0) {
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	const vector<int> labels = rhs.getLabelOfNode(pixel);
	for (vector<int>::const_iterator label_it = labels.begin(); label_it != labels.end(); label_it++) {
	  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	    int segment_id = *label_it / static_cast<int>(pow(RHS_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index)) % (RHS_NUM_SEGMENTS + 1);
	    if (segment_id == RHS_NUM_SEGMENTS)
	      continue;
	    rhs_unique_segment_ids.insert(segment_id);
	  }
	}
	if (rhs_unique_segment_ids.size() == RHS_NUM_SEGMENTS)
	  break;
      }
    }
  }
  const int UNION_NUM_SEGMENTS = unique_segment_ids.size() + rhs_unique_segment_ids.size();
  
  
  int new_segment_id = 0;
  map<int, Segment> union_segments;
  vector<vector<int> > union_labels(NUM_PIXELS_);
  {
    map<int, int> segment_id_map;
    const int NUM_SEGMENTS = getNumSegments();
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      const vector<int> labels = getLabelOfNode(pixel);
      for (vector<int>::const_iterator label_it = labels.begin(); label_it != labels.end(); label_it++) {
	for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	  int segment_id = *label_it / static_cast<int>(pow(NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index)) % (NUM_SEGMENTS + 1);
	  if (segment_id == NUM_SEGMENTS)
	    continue;
	  if (segment_id_map.count(segment_id) == 0) {
	    segment_id_map[segment_id] = new_segment_id;
	    new_segment_id++;
	  }
	}
      }
      if (segment_id_map.size() == NUM_SEGMENTS)
	break;
    }
    segment_id_map[NUM_SEGMENTS] = UNION_NUM_SEGMENTS;
    
    for (map<int, Segment>::const_iterator segment_it = segments_.begin(); segment_it != segments_.end(); segment_it++) {
      if (segment_id_map.count(segment_it->first) > 0) {
        union_segments[segment_id_map[segment_it->first]] = segment_it->second;
      }
    }

    if (NUM_SEGMENTS > 0) {
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	const vector<int> labels = getLabelOfNode(pixel);
	for (vector<int>::const_iterator label_it = labels.begin(); label_it != labels.end(); label_it++) {
	  int new_label = 0;
	  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	    int segment_id = *label_it / static_cast<int>(pow(NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index)) % (NUM_SEGMENTS + 1);
	    new_label += segment_id_map[segment_id] * pow(UNION_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index);
	  }
	  union_labels[pixel].push_back(new_label);
	}
      }
    }

    // if (segments_.size() > 0)
    //   for (map<int, int>::const_iterator segment_it = segment_id_map.begin(); segment_it != segment_id_map.end(); segment_it++)
    //     cout << "self: " << segment_it->first << '\t' << segment_it->second << endl;
  }

  {
    map<int, int> rhs_segment_id_map;
    const int RHS_NUM_SEGMENTS = rhs.getNumSegments();
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      const vector<int> labels = rhs.getLabelOfNode(pixel);
      for (vector<int>::const_iterator label_it = labels.begin(); label_it != labels.end(); label_it++) {
	for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	  int segment_id = *label_it / static_cast<int>(pow(RHS_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index)) % (RHS_NUM_SEGMENTS + 1);
	  if (segment_id == RHS_NUM_SEGMENTS)
	    continue;
	  if (rhs_segment_id_map.count(segment_id) == 0) {
	    rhs_segment_id_map[segment_id] = new_segment_id;
	    new_segment_id++;
	  }
	}
      }
      if (rhs_segment_id_map.size() == RHS_NUM_SEGMENTS)
	break;
    }
    rhs_segment_id_map[RHS_NUM_SEGMENTS] = UNION_NUM_SEGMENTS;

    for (map<int, Segment>::const_iterator rhs_segment_it = rhs_segments.begin(); rhs_segment_it != rhs_segments.end(); rhs_segment_it++) {
      if (rhs_segment_id_map.count(rhs_segment_it->first) > 0) {
        union_segments[rhs_segment_id_map[rhs_segment_it->first]] = rhs_segment_it->second;
      }
    }

    if (RHS_NUM_SEGMENTS > 0) {
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        const vector<int> rhs_labels = rhs.getLabelOfNode(pixel);
	for (vector<int>::const_iterator rhs_label_it = rhs_labels.begin(); rhs_label_it != rhs_labels.end(); rhs_label_it++) {
	  int rhs_new_label = 0;
	  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	    int segment_id = *rhs_label_it / static_cast<int>(pow(RHS_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index)) % (RHS_NUM_SEGMENTS + 1);
	    rhs_new_label += rhs_segment_id_map[segment_id] * pow(UNION_NUM_SEGMENTS + 1, NUM_LAYERS_ - 1 - layer_index);
	  }
	  union_labels[pixel].push_back(rhs_new_label);
	}
      }
    }

    // if (rhs_segments.size() > 0)
    //   for (map<int, int>::const_iterator rhs_segment_it = rhs_segment_id_map.begin(); rhs_segment_it != rhs_segment_id_map.end(); rhs_segment_it++)
    // 	cout << "rhs: " << rhs_segment_it->first << '\t' << rhs_segment_it->second << endl;
  }

  if (UNION_NUM_SEGMENTS == 0)
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      if (union_labels[pixel].size() == 0)
        union_labels[pixel].push_back(0);

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
