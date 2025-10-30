#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

class Node {
public:
  float data;
  Node *next;
  Node *prev;
  Node(float value) : data(value), next(nullptr), prev(nullptr) {}
};

class NeuralNetwork {

private:
  Node *input1;
  Node *input2;

public:
  NeuralNetwork() {
    input1 = nullptr;
    input2 = nullptr;
  }

  Node *MakeNode() {
    Node *newNode = new Node(0.5);
    return newNode;
  }

  void JoinNodes(Node *parent, Node *child) {
    parent->next = child;
    child->prev = parent;
  }

  void builder() {
    // First level of nodes -- making new nodes
    Node *lvl1Node1 = MakeNode();
    Node *lvl1Node2 = MakeNode();
    Node *lvl1Node3 = MakeNode();

    // First level -- joing other parent
    JoinNodes(input1, lvl1Node1);
    JoinNodes(input1, lvl1Node2);
    JoinNodes(input1, lvl1Node3);

    JoinNodes(input2, lvl1Node3);
    JoinNodes(input2, lvl1Node3);
    JoinNodes(input2, lvl1Node3);

    // Second level of nodes -- new nodes
    Node *lvl2Node1 = MakeNode();
    Node *lvl2Node2 = MakeNode();

    // Second level - joing others
    JoinNodes(lvl1Node1, lvl2Node1);
    JoinNodes(lvl1Node1, lvl2Node2);
    JoinNodes(lvl1Node2, lvl2Node1);
    JoinNodes(lvl1Node2, lvl2Node2);
    JoinNodes(lvl1Node3, lvl2Node1);
    JoinNodes(lvl1Node3, lvl2Node2);

    // Output node
    Node *output = MakeNode();

    // Output - join level2
    JoinNodes(lvl2Node1, output);
    JoinNodes(lvl2Node2, output);
  }
};

int main() {
  // Node n(10);
  // cout << "Node data: " << n.data << endl;
  //
  //

  NeuralNetwork nn; // = new NeuralNetwork();

  nn.builder();

  return 0;
}
