#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

class Node;

class Link {
public:
  Link *next;

  Node *prevNode;
  Node *nextNode;

  Link() : next(nullptr) {}
  Link(Node *prevNode, Node *nextNode)
      : prevNode(prevNode), nextNode(nextNode) {}
};

// linkedlist of links
class Join {
private:
  Link *head;

public:
  Join() { head = nullptr; }

  void AddConnection(Node *prev, Node *next) {
    Link *newLink = new Link(prev, next);

    if (head == nullptr) {
      //
      head = nullptr;
    } else {
      Link *temp = head;

      while (temp->next != nullptr) {
        temp = temp->next;
      }
      temp->next = newLink;
    }
  }

  Node *IterateJoins() {
    Link *temp = head;

    return temp->next->nextNode;

    // tester
    //
    /*while (temp != nullptr) {*/
    /**/
    /*  temp = temp->next;*/
    /**/
    /*  return temp->nextNode;*/
    /*}*/
    /*return nullptr;*/
  }
};

// every node have a data point and a list of all its connections
class Node {
public:
  float data;

  // Two simple linkedlist of nexts and prevs
  Join joins;

  Node(float value) : data(value) {}
  Node(Node *prev, Node *next) {}

  void AddConnection(Node *prev, Node *next) {
    //
    joins.AddConnection(prev, next);
  }

  Node *IterateJoins() {
    //
    return joins.IterateJoins();
  }
};

class NeuralNetwork {

private:
  Node *input1;
  Node *input2;

public:
  NeuralNetwork() {
    // input1 = nullptr;
    // input2 = nullptr;
  }

  Node *MakeNode(float value) {
    Node *newNode = new Node(value);
    return newNode;
  }

  void builder() {

    input1 = MakeNode(1.2);
    input2 = MakeNode(1.1);

    // First level of nodes -- making new nodes
    Node *lvl1Node1 = MakeNode(1.0);
    Node *lvl1Node2 = MakeNode(0.9);
    Node *lvl1Node3 = MakeNode(0.8);

    Node *output1 = MakeNode(0.7);
    Node *output2 = MakeNode(0.6);

    input1->AddConnection(nullptr, lvl1Node1);
    input1->AddConnection(nullptr, lvl1Node2);
    input1->AddConnection(nullptr, lvl1Node3);

    input2->AddConnection(nullptr, lvl1Node1);
    input2->AddConnection(nullptr, lvl1Node2);
    input2->AddConnection(nullptr, lvl1Node3);

    lvl1Node1->AddConnection(input1, nullptr);
    lvl1Node2->AddConnection(input1, nullptr);
    lvl1Node3->AddConnection(input1, nullptr);

    lvl1Node1->AddConnection(input2, nullptr);
    lvl1Node2->AddConnection(input2, nullptr);
    lvl1Node3->AddConnection(input2, nullptr);

    output1->AddConnection(lvl1Node1, nullptr);
    output1->AddConnection(lvl1Node2, nullptr);
    output1->AddConnection(lvl1Node3, nullptr);

    output2->AddConnection(lvl1Node1, nullptr);
    output2->AddConnection(lvl1Node2, nullptr);
    output2->AddConnection(lvl1Node3, nullptr);
  }

  void iterate() {

    Node *temp = input1;
    cout << temp->data;

    cout << temp->IterateJoins()->data;

    // temp->joins.IterateJoins();
  }
};

int main() {
  // Node n(10);
  // cout << "Node data: " << n.data << endl;
  //
  //

  NeuralNetwork nn; // = new NeuralNetwork();

  nn.builder();
  nn.iterate();

  return 0;
}
