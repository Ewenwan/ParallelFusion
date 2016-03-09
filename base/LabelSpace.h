#ifndef LABEL_SPACE_H__
#define LABEL_SPACE_H__

#include <vector>
#include <glog/logging.h>
#include <algorithm>
#include <memory>

namespace ParallelFusion {
//Label space management
    template<typename LabelType>
    class LabelSpace {
    public:
        LabelSpace(): num_nodes_(0) { };

        LabelSpace(const int NUM_NODES) : num_nodes_(NUM_NODES), label_space_((size_t)NUM_NODES) { };

        LabelSpace(const std::vector<LabelType> &single_solution);

        LabelSpace(const std::vector<std::vector<LabelType> > &label_space);

        void clear();

        inline void init(const int NUM_NODE, std::vector<int> v = std::vector<int>()){
            num_nodes_ = NUM_NODE;
            label_space_.resize((size_t)NUM_NODE, v);
        }

        inline int getNumNode() const{
            return num_nodes_;
        }

        inline std::vector<std::vector<LabelType> >& getLabelSpace(){
            return label_space_;
        }
        inline const std::vector<std::vector<LabelType> >& getLabelSpace() const{
            return label_space_;
        }

        inline bool empty() const{ return label_space_.empty();}

        inline const std::vector<LabelType>& getLabelOfNode(const int nid) const{
            CHECK_GE(nid, 0);
            CHECK_LT(nid, label_space_.size());
            return label_space_[nid];
        }

        inline std::vector<LabelType>& getLabelOfNode(const int nid){
            CHECK_GE(nid, 0);
            CHECK_LT(nid, label_space_.size());
            return label_space_[nid];
        }

        void assign(const int NUM_NODES, const std::vector<LabelType> &node_labels = std::vector<LabelType>()) {
            num_nodes_ = NUM_NODES;
            label_space_.assign(NUM_NODES, node_labels);
        };

        void appendSpace(const LabelSpace<LabelType> & rhs);
        void unionSpace(const LabelSpace<LabelType> &rhs);

        void setSingleLabels(const std::vector<LabelType> &single_labels);

        void setLabelSpace(const std::vector<std::vector<LabelType> > &label_space) {
            num_nodes_ = (int)label_space.size();
            label_space_ = label_space;
        };


        LabelType operator()(const int nodeid, const int sid) const{
            CHECK_LT(nodeid, label_space_.size());
            CHECK_LT(sid, label_space_[nodeid].size());
            return label_space_[nodeid][sid];
        }

        LabelType& operator()(const int nodeid, const int sid){
            CHECK_LT(nodeid, label_space_.size());
            CHECK_LT(sid, label_space_[nodeid].size());
            return label_space_[nodeid][sid];
        }

    protected:
        int num_nodes_;
        std::vector<std::vector<LabelType> > label_space_;
    };

    template<typename LabelType>
      LabelSpace<LabelType>::LabelSpace(const std::vector<LabelType> &single_labels) : num_nodes_((int)single_labels.size()) {
        label_space_.assign(num_nodes_, std::vector<LabelType>());
        setSingleLabels(single_labels);
    }

    template<typename LabelType>
    LabelSpace<LabelType>::LabelSpace(const std::vector<std::vector<LabelType> > &label_space) : num_nodes_((int)label_space.size()), label_space_(label_space) {
    }

    template<typename LabelType>
    void LabelSpace<LabelType>::clear() {
        for (int node_index = 0; node_index < num_nodes_; node_index++)
            label_space_[node_index].clear();
    }

    template<typename LabelType>
    void LabelSpace<LabelType>::setSingleLabels(const std::vector<LabelType> &single_labels) {
        //CHECK(single_labels.size() == num_nodes_) << "The number of nodes is inconsistent.";
        num_nodes_ = (int)single_labels.size();
        for (int node_index = 0; node_index < num_nodes_; node_index++)
            label_space_[node_index] = std::vector<LabelType>(1, single_labels[node_index]);
    }


    template<typename LabelType>
    void LabelSpace<LabelType>::appendSpace(const LabelSpace<LabelType> &rhs) {
        if(num_nodes_ == 0) {
            init(rhs.getNumNode());
        }
        for(auto i=0; i<rhs.getNumNode(); ++i){
            for(auto j=0; j<rhs.getLabelOfNode(i).size(); ++j)
                label_space_[i].push_back(rhs(i,j));
        }
    }

    template<typename LabelType>
    void LabelSpace<LabelType>::unionSpace(const LabelSpace<LabelType> &rhs){
        const std::vector<std::vector<LabelType> >& rhs_label_space = rhs.getLabelSpace();
        CHECK_EQ(label_space_.size(), rhs_label_space.size());
        if (num_nodes_ == 0) {
            num_nodes_ = (int)rhs_label_space.size();
            label_space_ = rhs_label_space;
        } else {
            for (int node_index = 0; node_index < num_nodes_; node_index++) {
                std::vector<LabelType> node_labels = label_space_[node_index];
                std::vector<LabelType> rhs_node_labels = rhs_label_space[node_index];
                //sort(node_labels.begin(), node_labels.end());    Maybe we assume?
                //sort(rhs_node_labels.begin(), rhs_node_labels.end());
                std::vector<LabelType> union_node_labels(node_labels.size() + rhs_node_labels.size());
                typename std::vector<LabelType>::const_iterator union_node_labels_it = std::set_union(node_labels.begin(),
                                                                                                      node_labels.end(),
                                                                                                      rhs_node_labels.begin(),
                                                                                                      rhs_node_labels.end(),
                                                                                                      union_node_labels.begin());
                union_node_labels.resize(union_node_labels_it - union_node_labels.begin());
                label_space_[node_index] = union_node_labels;
            }
        }
    }
}//namespace ParallelFusion
#endif
