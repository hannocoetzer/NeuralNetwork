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

//todo
// Error calculation
// f'(x)

/*
extra tools
https://www.tinkershop.net/ml/sigmoid_calculator.html

Activation functions (used to populate the network's input/values)
https://www.youtube.com/watch?v=LW9VnXVIQt4&list=PLLV9F4cTjkGMPeII0bxTgbF3GMiVPXry2&index=2 @6:50
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
- (-1)xError x f'(activation/sigmoid fuction) -- Note the S'(X) is the value to use when using the sigmoid calculator
-We save this value above in the node
-And then calculate and save the gradient in the weight by multiplying the 1st Node x Linked node value 

Weight calculations AKA back propagation / resilient propagation ( used to determine the weight adjustments)
https://youtu.be/IruMm7mPDdM?si=LogrtdF721h1wstU
- The weight values can be quite large
- each weight is adjusted by 
- network is populated with random weights initially
- New Weight = Current Weight + [Learn Rate (constant of 0.7) x Gradient] + [Momentum (constant of 0.3) x Previous Weight adjustment(previous iteration)]

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

  Node(float value) : data(value) {
    Sum = 0;
  }
  void addNext(Link *_link) {
    nexts.push_back(_link);
  }

  void addPrev(Link *_link){
    prevs.push_back(_link);
  }
};

class NeuralNetwork {

private:

  list<list<Node *>> layers;
  list<Node *> tempLayer;



public:
  NeuralNetwork() {

  }

  void initLayer()
  {
    //ensures we don't add prevs to the input nodes
    if(!layers.empty())
    {
      for(Node* prev : layers.back())
      {      
        for(Node* next : tempLayer)
        {
          prev->addNext(new Link(next,2));
          next->addPrev(new Link(prev,3));
        }
      }
    }

    layers.push_back(tempLayer);   
    tempLayer.clear(); 
  }

  Node *newNode(float value) 
  {
    Node *newNode = new Node(value);
    tempLayer.push_back(newNode);
    return newNode; 
  }

  void builder() {

    Node *input1 = newNode(0.1);
    Node *input2 = newNode(0.2);
    initLayer();
    
    Node *lvl1Node1 = newNode(0.3);
    Node *lvl1Node2 = newNode(0.4);
    Node *lvl1Node3 = newNode(0.5);
    initLayer();

    Node *output1 = newNode(0.6);
    Node *output2 = newNode(0.7);
    initLayer();

    displayNodePrevs(output1);    

    breathFirst();

    displayNodePrevs(output1);
  }

  void displayNodeNexts(Node* node)
  {
    for(Link* link: node->nexts)
    {
      cout<<endl<<link->node->data;
    }
  }

  void displayNodePrevs(Node* node)
  {
    for(Link* link: node->prevs)
    {
      cout<<endl<<"Sum : " <<link->node->Sum;
      cout<<endl<<"Weight : " <<link->weight;
      cout<<endl<<"In/output " <<link->node->data;
    }
  }
  
  void breathFirst(){
    int layerLevel = 1;
    for(list layer : layers)
    {
      if(!layerLevel == 1)
      {
        for(Node* layerNode : layer)
        {
          float sum = 0;
          //calculate sum
          for(Link *link : layerNode->prevs)
          {
            sum = sum + (link->weight) * (link->node->data);
            cout<<endl<<sum;
          }
          layerNode->Sum = sum;
          //do sigma
          layerNode->data = sigma(sum);
        }
      }
      layerLevel++;
    }
  }

  float sigma(float x){
    //faster as mentioned here https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm 
    //return value / (1 + abs(value));

    //f=1/(1+e^(-x)) 
    const float e = 2.71828;
    return 1/(1+pow(e,x));
  }

  float sigmaDerivative(float x){
    //f'=(e^x)/[e^(2x)+2e^(x)+1] 
    const float e = 2.71828;
    return pow(e,x) / (pow(e,2*x) + 2*pow(e,x) + 1 );

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
