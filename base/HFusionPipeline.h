//
// Created by yanhang on 3/9/16.
//

#ifndef PARALLELFUSION_HFUSIONPIPELINE_H
#define PARALLELFUSION_HFUSIONPIPELINE_H
#include "LabelSpace.h"
#include "ProposalGenerator.h"
#include "FusionSolver.h"
#include "pipeline_util.h"

namespace ParallelFusion{

    template<typename T>
    class BinaryTree{
    public:
        struct Node{
            Node(): data(NULL), lchild(NULL), rchild(NULL), parent(NULL), flag(false){}
            std::shared_ptr<T> data;
            std::shared_ptr<Node> lchild;
            std::shared_ptr<Node> rchild;
            std::shared_ptr<Node> parent;
            //in HFusion, flag indicates whether this node contains data
            bool flag;
        };

        BinaryTree(const BinaryTree& t2){
            root = t2.getRoot();
        }

        BinaryTree(const std::shared_ptr<Node> r2){
            root = r2;
        }

        void buildFromLeafData(const std::vector<T>& leafData);

        void buildFromLeafNodes(const std::vector<std::shared_ptr<Node> >& leafNodes);

        inline const std::shared_ptr<Node> getRoot()const{
            return root;
        }
    private:
        std::shared_ptr<Node> root;
    };


    struct HFusionPipelineOption{
    public:
        HFusionPipelineOption(): num_threads(4){}
        int num_threads;
    };

    template<class LABELSPACE>
    class HFusionPipeline{
    public:
        HFusionPipeline(const HFusionPipelineOption& option_): option(option_){}

    private:
        HFusionPipelineOption option;
    };

    /////////////////////////////////////////////////////////////
    template<typename T>
    void BinaryTree<T>::buildFromLeafData(const std::vector<T>& leafData){
        std::vector<std::shared_ptr<Node> > leafNodes(leafData.size(), new Node());
        for(auto i=0; i<leafData; ++i){
            leafNodes[i].reset();
        }
        buildFromLeafNodes(leafNodes);
    }

    template<typename T>
    void BinaryTree<T>::buildFromLeafNodes(const std::vector<std::shared_ptr<Node> >& leafNodes) {
        std::vector<Node> tree = leafNodes;
        while(tree.size() >= 2){
            std::vector<Node> cachedNode;
            for(auto i=0; i< tree.size(); i += 2){
                if(i < tree.size() - 1){
                    Node newNode;
                    newNode.parent =
                }
            }
        }
    }
}//namespace ParallelFusion

#endif //PARALLELFUSION_HFUSIONPIPELINE_H