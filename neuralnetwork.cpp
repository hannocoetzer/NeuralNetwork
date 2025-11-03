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

class Node;

class Link {

public:
  Node *next;
  float weight;

  Link(Node *next, float weight) : next(next), weight(weight) {}
};

class Node {
public:
  float data;
  list<Link *> nexts;

  Node(float value) : data(value) {}
  Link *AddLink(Node *next, float weight) {

    Link *newLink = new Link(next, weight);
    nexts.push_back(newLink);
    return newLink;
  }
};

class NeuralNetwork {

private:
  Node *input1;
  Node *input2;
  list<Node *> layer1;
  list<Link *> linksToLayer1;

public:
  NeuralNetwork() {
    input1 = nullptr;
    input2 = nullptr;
  }

  Node *MakeNode(float value) { return new Node(value); }

  void builder() {

    int sizeOfLayer1 = 3;

    /*input1 = MakeNode(1.0);*/
    /*input2 = MakeNode(1.0);*/
    /**/
    /*Node *lvl1Node1 = MakeNode(0.9);*/
    /*Node *lvl1Node2 = MakeNode(0.8);*/
    /*Node *lvl1Node3 = MakeNode(0.7);*/
    /**/
    /*Node *output1 = MakeNode(0.1);*/
    /*Node *output2 = MakeNode(0.2);*/
    /**/
    /*input1->AddLink(lvl1Node1, 0.6);*/
    /*input1->AddLink(lvl1Node2, 0.5);*/
    /*input1->AddLink(lvl1Node3, 0.4);*/
    /**/
    /*input2->AddLink(lvl1Node1, 0.62);*/
    /*input2->AddLink(lvl1Node2, 0.52);*/
    /*input2->AddLink(lvl1Node3, 0.42);*/
    /**/
    /*lvl1Node1->AddLink(output1, 0.31);*/
    /*lvl1Node2->AddLink(output1, 0.21);*/
    /*lvl1Node3->AddLink(output1, 0.11);*/
    /**/
    /*lvl1Node1->AddLink(output2, 0.32);*/
    /*lvl1Node2->AddLink(output2, 0.22);*/
    /*lvl1Node3->AddLink(output2, 0.12);*/

    Iterator(input1);
  }

  void Iterator(Node *tmp) {

    cout << endl << tmp->data;
    if (!tmp->nexts.empty()) {

      for (Link *link : tmp->nexts) {

        cout << endl << "weight " << link->weight << endl;

        Iterator(link->next);
      }
    }
  }
};

int main() {
  NeuralNetwork nn;
  nn.builder();

  return 0;
}
