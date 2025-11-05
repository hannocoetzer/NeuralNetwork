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

/*
Activation functions (used to populate the network with input/values | network is populated with random weights )
https://www.youtube.com/watch?v=LW9VnXVIQt4&list=PLLV9F4cTjkGMPeII0bxTgbF3GMiVPXry2&index=2 @6:50
- The weight values can be quite large
- Activation fx = sigmoid function[(I1xW1) + (I2 x W2) + (I3 x W3) + (In x Wn)] = Input to another Node
- where I is the input/value of the node and W is the weight
- this function makes it so that the output/value of the Node is in a range of -1 to 1

Error calculations (used to check if your output is within range)
https://www.youtube.com/watch?v=U4BTzF3Wzt0&list=PLLV9F4cTjkGNKcpD7PEFdYZvDP81xDT7w&index=3 @9:40
- Used to determine the error (or how close you are) to the actual output value it should be
- Mean square is commonly used
- [(I1-A1)^2 + (I2-A2)^2 + (I3-A3)^2 + (In-An)^2]/ n
- I is the Ideal and A is the Actual

Gradient calculations (first step into calculating the weights)
https://www.youtube.com/watch?v=p1-FiWjThs8&list=PLLV9F4cTjkGNKcpD7PEFdYZvDP81xDT7w&index=3 @ 4:52

Weight calculations AKA back propagation AKA resilient prop

https://youtu.be/IruMm7mPDdM?si=LogrtdF721h1wstU

*/



class Node;

class Link {

public:
  Node *next;
  string weight;

  Link(Node *next, string weight) : next(next), weight(weight) {}
};

class Node {
public:
  string data;
  list<Link *> nexts;

  Node(string value) : data(value) {}
  Link *AddLink(Node *next, string weight) {

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

  Node *MakeNode(string value) { return new Node(value); }

  void builder() {

    int sizeOfLayer1 = 3;

    input1 = MakeNode("A");
    input2 = MakeNode("B");
    
    Node *lvl1Node1 = MakeNode("C");
    Node *lvl1Node2 = MakeNode("D");
    Node *lvl1Node3 = MakeNode("E");
    Node *output1 = MakeNode("F");
    Node *output2 = MakeNode("G");
    
    input1->AddLink(lvl1Node1, "A-C");
    input1->AddLink(lvl1Node2, "A-D");
    input1->AddLink(lvl1Node3, "A-E");
    
    input2->AddLink(lvl1Node1, "B-C");
    input2->AddLink(lvl1Node2, "B-D");
    input2->AddLink(lvl1Node3, "B-E");
    
    lvl1Node1->AddLink(output1, "C-F");
    lvl1Node2->AddLink(output1, "D-F");
    lvl1Node3->AddLink(output1, "E-F");
    
    lvl1Node1->AddLink(output2, "C-G");
    lvl1Node2->AddLink(output2, "D-G");
    lvl1Node3->AddLink(output2, "E-G");

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
