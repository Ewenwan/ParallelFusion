//
// Created by Hang Yan on 3/10/16.
// Washington University in St. Louis
// yanhang@wustl.edu
//

#ifndef HANG_BINARY_TREE_H
#define HANG_BINARY_TREE_H

#include <vector>
#include <memory>
#include <functional>
#include <glog/logging.h>
#include <atomic>

namespace Hang_BinaryTree {
    template<typename T>
    struct Node {
        Node():lchild(NULL), rchild(NULL), parent(NULL), depth(-1), nodeId(-1), flag1(false), flag2(false) { }

        T data;
        std::shared_ptr<Node> lchild;
        std::shared_ptr<Node> rchild;
        std::shared_ptr<Node> parent;
        int depth;
        int nodeId;

        //two auxilary flags
        std::atomic<bool> flag1;
        std::atomic<bool> flag2;
    };

    enum TraverseMethod {
        PRE_ORDER, IN_ORDER, POST_ORDER
    };

    template<typename T>
    class BinaryTree {
    public:
        typedef Node<T> NodeT;
        typedef std::function<void(std::shared_ptr<NodeT>)> FunctorType;
        BinaryTree(): depth(-1), numNode(0){}

        BinaryTree(const BinaryTree &t2) {
            BinaryTree();
            root = t2.getRoot();
        }

        BinaryTree(const std::shared_ptr<NodeT> r2) {
            BinaryTree();
            root = r2;
        }

        void buildFromLeafData(const std::vector<T> &leafData);

        //method of tree traversal.
        //input: method: pre-order, in-order or post-order
        //option: callable object represents what options should be perform on each node.
        //        This can be a struct or class with operator(), or a lambda function.
        //        Notice that this operation can not have return value.
        void traverseTree(std::function<void(std::shared_ptr<NodeT>)> &operation, const TraverseMethod method,
                          std::shared_ptr<NodeT> startNode);

        inline const std::shared_ptr<NodeT> getRoot() const {
            return root;
        }

        const inline int getTreeDepth() const{
            return depth;
        }

        const inline int getNumNode() const{
            return numNode;
        }

    private:
        void assignNodeDepth();
        void assignNodeId();

        std::shared_ptr<NodeT> root;
        int depth;
        int numNode;
    };

/////////////////////////////////////////////////////////////
    template<typename T>
    void BinaryTree<T>::buildFromLeafData(const std::vector<T> &leafData) {
        std::vector<std::shared_ptr<NodeT> > tree(leafData.size());
        for(auto i=0; i<tree.size(); ++i){
            tree[i] = std::shared_ptr<NodeT>(new NodeT());
            tree[i]->data = leafData[i];
        }
        numNode = (int)tree.size();
        while (tree.size() > 1) {
            std::vector<std::shared_ptr<NodeT> > cachedNode;
            for (auto i = 0; i < tree.size(); i += 2) {
                if (i < tree.size() - 1) {
                    std::shared_ptr<NodeT> curNode(new NodeT());
                    curNode->lchild = tree[i];
                    curNode->rchild = tree[i + 1];
                    curNode->lchild->parent = curNode;
                    curNode->rchild->parent = curNode;
                    cachedNode.push_back(curNode);
                    numNode++;
                } else {
                    cachedNode.push_back(tree[i]);
                }
            }
            tree.swap(cachedNode);
        }

        CHECK_EQ(tree.size(), 1);
        root = tree[0];

        assignNodeDepth();
        assignNodeId();
    }

    template<typename T>
    void BinaryTree<T>::traverseTree(std::function<void(std::shared_ptr<NodeT>)> &operation, const TraverseMethod method,
                                     std::shared_ptr<NodeT> startNode) {
        if(!startNode.get())
            return;
        switch (method) {
            case PRE_ORDER:
                operation(startNode);
                traverseTree(operation, method, startNode->lchild);
                traverseTree(operation, method, startNode->rchild);
                break;
            case IN_ORDER:
                traverseTree(operation, method, startNode->lchild);
                operation(startNode);
                traverseTree(operation, method, startNode->rchild);
                break;
            case POST_ORDER:
                traverseTree(operation, method, startNode->lchild);
                traverseTree(operation, method, startNode->rchild);
                operation(startNode);
                break;
            default:
                break;
        }
    }

    template<typename T>
    void BinaryTree<T>::assignNodeDepth() {
         FunctorType f = [](std::shared_ptr<NodeT> node){
            if(node->parent.get())
                node->depth = node->parent->depth + 1;
            else
                node->depth = 0;
        };
        traverseTree(f, PRE_ORDER, getRoot());
    }

    template<typename T>
    void BinaryTree<T>::assignNodeId() {
        int startId = 0;
        FunctorType f = [&](std::shared_ptr<NodeT> node){
            node->nodeId = startId++;
        };
        traverseTree(f, PRE_ORDER, getRoot());
    }
}//namespace Hang_BinaryTree
#endif //HANG_BINARY_TREE_H
