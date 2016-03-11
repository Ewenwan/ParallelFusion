//
// Created by yanhang on 3/7/16.
//

#include "FusionSolver.h"
#include "HFusionPipeline.h"
#include <iostream>

using namespace std;
using namespace ParallelFusion;

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    //test for binary tree
    Hang_BinaryTree::BinaryTree<int> btree;
    vector<int> leaf{1,2,3,4,5,6,7,8,9,10};

    cout << "Building tree" << endl << flush;
    btree.buildFromLeafData(leaf);
    cout << "done. Number of nodes: " << btree.getNumNode() <<  endl << flush;

    Hang_BinaryTree::BinaryTree<int>::FunctorType traversePrint = [](shared_ptr<Hang_BinaryTree::Node<int> > node){
        printf("(%d,%d) ", node->nodeId, node->data);
    };

    cout << "Preorder:" << endl;
    btree.traverseTree(traversePrint, Hang_BinaryTree::PRE_ORDER, btree.getRoot());
    cout << endl;
    cout << "Inorder:" << endl;
    btree.traverseTree(traversePrint, Hang_BinaryTree::IN_ORDER, btree.getRoot());
    cout << endl;
    cout << "Posteorder:" << endl;
    btree.traverseTree(traversePrint, Hang_BinaryTree::POST_ORDER, btree.getRoot());
    cout << endl;
    return 0;
}