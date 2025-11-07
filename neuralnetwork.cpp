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
extra tools
https://www.tinkershop.net/ml/sigmoid_calculator.html

Activation functions (used to populate the network's input/values | network is populated with random weights initially I think)
https://www.youtube.com/watch?v=LW9VnXVIQt4&list=PLLV9F4cTjkGMPeII0bxTgbF3GMiVPXry2&index=2 @6:50
- The weight values can be quite large
- Activation fx = sigmoid function[(I1xW1) + (I2 x W2) + (I3 x W3) + (In x Wn)] = Input to another Node
- where I is the input/value of the node and W is the weight
- this function makes it so that the output/value of the Node is in a range of -1 to 1

Error calculations (used to check if your output is within range, or close to the training output )
https://www.youtube.com/watch?v=U4BTzF3Wzt0&list=PLLV9F4cTjkGNKcpD7PEFdYZvDP81xDT7w&index=3 @9:40
- Used to determine the error (or how close you are) to the actual output value it should be
- Mean square is commonly used
- [(I1-A1)^2 + (I2-A2)^2 + (I3-A3)^2 + (In-An)^2]/ n
- I is the Ideal and A is the Actual

Gradient calculations used/saved in the weights (first step into calculating the weight adjustments for the network)
https://www.youtube.com/watch?v=p1-FiWjThs8&list=PLLV9F4cTjkGNKcpD7PEFdYZvDP81xDT7w&index=3 @ 4:52
- To be corrected but roughly calculated as
  - (-1)xError x f'(activation/sigma value) -- Note the S'(X) is the value to use when using the sigmoid calculator
  -We save this value above in the node
  -And then calculate and save the gradient in the weight by multiplying the 1st Node x Linked node value 

Weight calculations AKA back propagation / resilient propagation ( used to determine the weight adjustments)
https://youtu.be/IruMm7mPDdM?si=LogrtdF721h1wstU
- each weight is adjusted by 
New Weight = Current Weight + [Learn Rate (constant of 0.7) x Gradient] + [Momentum (constant of 0.3) x Previous Weight adjustment(0 for first iteration)]

more advanced (ADAM)
https://www.youtube.com/watch?v=zUZRUTJbYm8

*/



class Node;

class Link {

public:
  Node *node;
  float weight;
  float gradient; // this value is the (-1) x Error x Derivative( Activation func ) !! Activation func = SIGMOID(Sum)

  Link( Node *_link, float _weight) :  node(_link), weight(_weight) {}
};

class Node {
public:
  float data;
  float Sum;  // this value is the SUM OF ALL [linked(parent) node values x weights] 
  list<Link *> nexts;
  list<Link *> prevs;

  Node(float value) : data(value) {}
  void AddNext(Link *_link) {
    nexts.push_back(_link);
  }

  void AddPrev(Link *_link){
    prevs.push_back(_link);
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


    input1 = MakeNode(0.1);
    input2 = MakeNode(0.2);
    
    Node *lvl1Node1 = MakeNode(0.3);
    Node *lvl1Node2 = MakeNode(0.4);
    Node *lvl1Node3 = MakeNode(0.5);
    Node *output1 = MakeNode(0.6);
    Node *output2 = MakeNode(0.7);

    input1->AddNext(new Link(lvl1Node1,0.13));    
    input1->AddNext(new Link(lvl1Node2,0.14));
    input1->AddNext(new Link(lvl1Node3,0.15));

    lvl1Node1->AddPrev(new Link(input1,0.13));    
    lvl1Node2->AddPrev(new Link(input1,0.14));
    lvl1Node3->AddPrev(new Link(input1,0.15));

    input2->AddNext(new Link(lvl1Node1,0.23));    
    input2->AddNext(new Link(lvl1Node2,0.24));
    input2->AddNext(new Link(lvl1Node3,0.25));

    lvl1Node1->AddPrev(new Link(input2,0.23));    
    lvl1Node2->AddPrev(new Link(input2,0.24));
    lvl1Node3->AddPrev(new Link(input2,0.25));
    
    
    lvl1Node1->AddNext(new Link(output1,0.36));    
    lvl1Node2->AddNext(new Link(output1,0.46));
    lvl1Node3->AddNext(new Link(output1,0.56));

    output1->AddPrev(new Link(lvl1Node1,0.36));    
    output1->AddPrev(new Link(lvl1Node2,0.46));
    output1->AddPrev(new Link(lvl1Node3,0.56));

    lvl1Node1->AddNext(new Link(output2,0.37));    
    lvl1Node2->AddNext(new Link(output2,0.47));
    lvl1Node3->AddNext(new Link(output2,0.57));

    output2->AddPrev(new Link(lvl1Node1,0.37));    
    output2->AddPrev(new Link(lvl1Node2,0.47));
    output2->AddPrev(new Link(lvl1Node3,0.57));  
    
    //Iterator(input1);
    //1IteratorReverse(output1);
  }


  void Iterator(Node *tmp) {

    cout << endl << tmp->data;
    if (!tmp->nexts.empty()) {

      for (Link *link : tmp->nexts) {

        cout << endl << "weight " << link->weight << endl;

        Iterator(link->node);
      }
    }
  }

  void IteratorReverse(Node *tmp) {

    cout << endl << tmp->data;
    if (!tmp->prevs.empty()) {

      for (Link *link : tmp->prevs) {

        cout << endl << "weight " << link->weight << endl;

        Iterator(link->node);
      }
    }
  }

};

int main() {
  NeuralNetwork nn;
  nn.builder();

  return 0;
}
