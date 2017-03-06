#pragma once

#include <algorithm>
#include <glog/logging.h>
#include <memory>
#include <vector>

namespace ParallelFusion {
// Label space management
template <class SPACE, class NODE, typename LABELTYPE> class LabelSpaceBase {
public:
  virtual void clear() = 0;
  virtual void init(const int NUM_NODE, NODE v) = 0;

  virtual int getNumNode() const = 0;

  virtual SPACE &getLabelSpace() = 0;
  virtual const SPACE &getLabelSpace() const = 0;

  virtual bool empty() const = 0;

  virtual const NODE &getLabelOfNode(const int nid) const = 0;
  virtual NODE &getLabelOfNode(const int nid) = 0;

  virtual void assign(const int NUM_NODES, const NODE &node_labels) = 0;
  virtual void
  appendSpace(const LabelSpaceBase<SPACE, NODE, LABELTYPE> &rhs) = 0;
  virtual void
  unionSpace(const LabelSpaceBase<SPACE, NODE, LABELTYPE> &rhs) = 0;

  virtual void setSingleLabels(const NODE &single_labels) = 0;

  virtual void setLabelSpace(const SPACE &label_space) = 0;

  virtual LABELTYPE &operator()(const int nodeid, const int sid) = 0;
  virtual const LABELTYPE &operator()(const int nodeid,
                                      const int sid) const = 0;
};

} // namespace ParallelFusion