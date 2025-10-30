#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <queue>
#include <stack>
using namespace std;

class Node {
public:
    float data;
    Node* next;
    Node(float value) : data(value), next(nullptr) {}
};

class NeuralNetwork{

    private:
        Node* input1;
        Node* input2;

    public:
        NeuralNetwork(){
            input1 = nullptr;
            input2 = nullptr;
        }

        Node* MakeNode()
        {
            Node* newNode = new Node(0.5);
            return newNode;
        }

        void JoinNodes(Node* parent,Node* child)
        {
           parent->next = child;
        }

        void builder()
        {
            //First level of nodes -- making new nodes
            Node* lvl1Node1 = MakeNode();
            Node* lvl1Node2 = MakeNode();
            Node* lvl1Node3 = MakeNode();

            //First level -- joing other parent
            JoinNodes(input1, lvl1Node1);
            JoinNodes(input1,lvl1Node2);
            JoinNodes(input1, lvl1Node3);

            JoinNodes(input2, lvl1Node3);
            JoinNodes(input2, lvl1Node3);
            JoinNodes(input2, lvl1Node3);

            //Second level of nodes -- new nodes
            Node* lvl2Node1 = MakeNode();
            Node* lvl2Node2 = MakeNode();

            //Second level - joing others
                       


        }

};

int main() {
    Node n(10);
    cout << "Node data: " << n.data << endl;
    return 0;
}