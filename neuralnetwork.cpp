#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
using namespace std;

class Node {
public:
  float data;
  list<Node *> prevs;
  list<Node *> nexts;

  Node(float value) : data(value) {}
  void AddPrev(Node *prev) { prevs.push_back(prev); }
  void AddNext(Node *next) { nexts.push_back(next); }
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

  Node *MakeNode(float value) { return new Node(value); }

  void builder() {

    input1 = MakeNode(1.2);
    input2 = MakeNode(1.1);

    // First level of nodes -- making new nodes
    Node *lvl1Node1 = MakeNode(1.01);
    Node *lvl1Node2 = MakeNode(0.9);
    Node *lvl1Node3 = MakeNode(0.8);

    Node *output1 = MakeNode(0.0);
    Node *output2 = MakeNode(0.0);

    input1->AddNext(lvl1Node1);
    input1->AddNext(lvl1Node2);
    input1->AddNext(lvl1Node3);

    input2->AddNext(lvl1Node1);
    input2->AddNext(lvl1Node2);
    input2->AddNext(lvl1Node3);

    lvl1Node1->AddPrev(input1);
    lvl1Node2->AddPrev(input1);
    lvl1Node3->AddPrev(input1);

    lvl1Node1->AddPrev(input2);
    lvl1Node2->AddPrev(input2);
    lvl1Node3->AddPrev(input2);

    lvl1Node1->AddNext(output1);
    lvl1Node2->AddNext(output1);
    lvl1Node3->AddNext(output1);

    lvl1Node1->AddNext(output2);
    lvl1Node2->AddNext(output2);
    lvl1Node3->AddNext(output2);

    output1->AddPrev(lvl1Node1);
    output1->AddPrev(lvl1Node2);
    output1->AddPrev(lvl1Node3);

    output2->AddPrev(lvl1Node1);
    output2->AddPrev(lvl1Node2);
    output2->AddPrev(lvl1Node3);

    Iterator(input1);
  }

  void Iterator(Node *tmp) {

    cout << tmp->data << endl;
    if (!tmp->nexts.empty()) {

      for (Node *node : tmp->nexts) {

        Iterator(node);
      }
    }
  }
};

int main() {
  NeuralNetwork nn;
  nn.builder();

  return 0;
}
