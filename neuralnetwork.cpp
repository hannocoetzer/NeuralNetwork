#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// todo
// ----
// depth first weird on sum and delta calculations
//  Error calculation
//  f'(x)

/*
extra tools
https://www.tinkershop.net/ml/sigmoid_calculator.html

Activation functions (used to populate the network's input/values)
https://www.youtube.com/watch?v=LW9VnXVIQt4&list=PLLV9F4cTjkGMPeII0bxTgbF3GMiVPXry2&index=2
@6:50
- Activation fx = sigmoid function[(I1xW1) + (I2 x W2) + (I3 x W3) + (In x Wn)]
= Input to another Node
- where I is the input/value of the node and W is the weight
- this function makes it so that the output/value of the Node is in a range of
-1 to 1

Error calculations (used to check if your output is within range, or close to
the training output )
https://www.youtube.com/watch?v=U4BTzF3Wzt0&list=PLLV9F4cTjkGNKcpD7PEFdYZvDP81xDT7w&index=2
@9:40
- Used to determine the error (or how close you are) to the actual output value
it should be
- Mean square is commonly used
- [(I1-A1)^2 + (I2-A2)^2 + (I3-A3)^2 + (In-An)^2]/ n
- I is the Ideal and A is the Actual

Gradient calculations used/saved in the weights (first step into calculating the
weight adjustments for the network)
https://www.youtube.com/watch?v=p1-FiWjThs8&list=PLLV9F4cTjkGNKcpD7PEFdYZvDP81xDT7w&index=3
@ 4:52
- To be corrected but roughly calculated as
- (-1)xError x f'(activation/sigmoid fuction) -- Note the S'(X) is the value to
use when using the sigmoid calculator -We save this value above in the node -And
then calculate and save the gradient in the weight by multiplying the 1st Node x
Linked node value
- we start at the last/output nodes and get the derivitavie value function of
the previous nodes connected to the output getting calculated (I1 x W1) + (I2 x
W2)

Weight calculations AKA back propagation / resilient propagation ( used to
determine the weight adjustments)
https://youtu.be/IruMm7mPDdM?si=LogrtdF721h1wstU
- The weight values can be quite large
- each weight is adjusted by
- network is populated with random weights initially
- New Weight = Current Weight + [Learn Rate (constant of 0.7) x Gradient] +
[Momentum (constant of 0.3) x Previous Weight adjustment(previous iteration)]

more advanced (ADAM)
https://www.youtube.com/watch?v=zUZRUTJbYm8

*/

enum NodeType
{
  INPUT,
  OUTPUT,
  BIAS,
  HIDDEN
};

class Node;

class Props
{
public:
  float weight;
  float momentum_multiplier;
  float gradient; // this value is the (-1) x Error x Derivative( Activation
                  // func ) !! Activation func = SIGMOID(Sum)
  Props(float _weight, float _gradient)
      : weight(_weight), gradient(_gradient) {
        momentum_multiplier = 0;

      }
};

class Link
{
public:
  Node *node;
  Props *props; // properties of this link : weight, gradient, etc
  Link(Node *_node, Props *_props) : node(_node), props(_props) {}
};

class Node
{
public:
  float data;
  float ideal;
  float delta;
  float sum; // this value is the SUM OF ALL [linked(parent) node values x weights]
  float dfSum;
  NodeType nodeType;

  list<Link *> nexts;
  list<Link *> prevs;

  Node(float value, NodeType _nodeType) : data(value)
  {
    sum = 0;
    delta = 0;
    dfSum = 0;
    nodeType = _nodeType;
  }
  Node(float value, float _ideal, NodeType _nodeType)
      : data(value), ideal(_ideal)
  {
    sum = 0;
    delta = 0;
    dfSum = 0;
    nodeType = _nodeType;
  }

  void addNext(Link *_link) { nexts.push_back(_link); }

  void addPrev(Link *_link) { prevs.push_back(_link); }
};

class NeuralNetwork
{
private:
  list<list<Node *>> layers;
  list<Node *> tempLayer;

public:
  NeuralNetwork() {}

  void initLayer()
  {
    // ensures we don't add prevs to the input nodes
    if (!layers.empty())
    {
      for (Node *prev : layers.back())
      {
        for (Node *next : tempLayer)
        {          
          //float r = rand() / (RAND_MAX + 1.);
          const int range_from  = -1000;
          const int range_to    = 1000;
          std::random_device                  rand_dev;
          std::mt19937                        generator(rand_dev());
          std::uniform_int_distribution<int>  distr(range_from, range_to);
          float r = (float)distr(generator)/1000;

          Props *newProps = new Props(r, 0);

          //connect all prev Nodes with next Nodes, except when next Node is a BIAS node
          if(!(next->nodeType == BIAS))
          {
            //if(!(next->nodeType == OUTPUT))
            prev->addNext(new Link(next, newProps));
            next->addPrev(new Link(prev, newProps));
          }

          // // nodes that are OUTPUT doesn't have next links
          // if (!(next->nodeType == OUTPUT))
          //   prev->addNext(new Link(next, newProps));

          // // nodes that is either BIAS nor INPUT nodes does not get to have prev links
          // else if (!(next->nodeType == BIAS) || !(next->nodeType == INPUT))
          //   next->addPrev(new Link(prev, newProps));
        }
      }
    }

    layers.push_back(tempLayer);
    tempLayer.clear();
  }

  Node *newNode(float value, NodeType _nodeType)
  {
    Node *newNode = new Node(value, _nodeType);
    tempLayer.push_back(newNode);
    return newNode;
  }

  Node *newNode(float value, float _ideal, NodeType _nodeType)
  {
    Node *newNode = new Node(value, _ideal, _nodeType);
    tempLayer.push_back(newNode);
    return newNode;
  }

  void tester(){
    Node *i1 = newNode(1, INPUT);
    Node *i2 = newNode(0, INPUT);
    Node *b1 = newNode(1,BIAS);
    i1->data = 1;

    layers.push_back(tempLayer);
    tempLayer.clear();

    Node *h1 = newNode(1, HIDDEN);
    h1->dfSum = sigmoidDerivative(-0.53);
    h1->data = 0.37;

    Node *h2 = newNode(1, HIDDEN);
    h2->dfSum = sigmoidDerivative(1.05);
    h2->data = 0.74;

    Node *b2 = newNode(1,BIAS);
    b2->data = 1;

    layers.push_back(tempLayer);
    tempLayer.clear();
    
    Node *o1 = newNode(1,0, OUTPUT);
    o1->dfSum = sigmoidDerivative(1.13);
    o1->data = 0.75;
    o1->ideal = 1;

    layers.push_back(tempLayer);
    tempLayer.clear();
    
    Props *i1h1 = new Props(-0.07, 0);
    i1->addNext(new Link(h1,i1h1));
    h1->addPrev(new Link(i1,i1h1));

    Props *i1h2 = new Props(0.94, 0);
    i1->addNext(new Link(h2,i1h2));
    h2->addPrev(new Link(i1,i1h2));

    Props *i2h1 = new Props(0.94, 0);
    i2->addNext(new Link(h1,i2h1));
    h1->addPrev(new Link(i2,i2h1));

    Props *i2h2 = new Props(0.46, 0);
    i2->addNext(new Link(h2,i2h2));
    h2->addPrev(new Link(i2,i2h2)); 

    Props *b1h1 = new Props(-0.46, 0);
    b1->addNext(new Link(h1,b1h1));
    h1->addPrev(new Link(b1,b1h1)); 

    Props *b1h2 = new Props(0.10, 0);
    b1->addNext(new Link(h2,b1h2));
    h2->addPrev(new Link(b1,b1h2)); 

    Props *h1o1 = new Props(-0.22, 0);
    h1->addNext(new Link(o1,h1o1));
    o1->addPrev(new Link(h1,h1o1)); 

    Props *h2o1 = new Props(0.58, 0);
    h2->addNext(new Link(o1,h2o1));
    o1->addPrev(new Link(h2,h2o1)); 

     Props *b2o1 = new Props(0.78, 0);
    b2->addNext(new Link(o1,b2o1));
    o1->addPrev(new Link(b2,b2o1));  
  
    depthFirst();
    adjustWeights(0.7,0.3);
  }

  void builder()
  {

    tester();

    Node *input1 = newNode(0, INPUT);
    Node *input2 = newNode(0, INPUT);
    Node *inputBias = newNode(1,BIAS);
    initLayer();

    Node *lvl1Node1 = newNode(1, HIDDEN);
    Node *lvl1Node2 = newNode(1, HIDDEN);
    Node *lvl1Bias2 = newNode(1,BIAS);
    initLayer();

    Node *output1 = newNode(1, OUTPUT);
    initLayer();

    float errorRate = error();
    float learnRate = 0.7;
    float momentumRate = 0.3;
    bool isTrainedWell = false;

    //online training
    input1->data = 1;
    input2->data = 0;
    output1->ideal = 1;
    while(!isTrainedWell)
    {
      breadthFirst();
      depthFirst();

      errorRate = error();
      cout<<endl<<errorRate;

      if(errorRate < 0.1)
      {
        isTrainedWell = true;
      }
      else{
        adjustWeights(learnRate,momentumRate);
      }
    }

    //batch training attempt
    //Note : check if error gets called
    /*int trainingsetNth = 0;
    vector<tuple<int,int,int>> trainingSet;
    trainingSet.push_back(make_tuple(0,0,0));
    trainingSet.push_back(make_tuple(0,1,1));
    trainingSet.push_back(make_tuple(1,0,1));
    trainingSet.push_back(make_tuple(1,1,0));
    while(!isTrainedWell)
    {
      breadthFirst();
      depthFirst();
      cout<<"ErrorRate : "<<errorRate<<endl;      
      adjustWeights(learnRate, momentumRate);      
      if(errorRate < 0.001)
      {
        input1->data = get<0>(trainingSet[trainingsetNth]);
        input2->data = get<1>(trainingSet[trainingsetNth]);
        output1->ideal = get<2>(trainingSet[trainingsetNth]);

        if(trainingsetNth % 3 == 0 ){    
          trainingsetNth ++;
          isTrainedWell = true;
        }
      }      
    }*/

    int testRuns = 5;
    while(testRuns > 0)
    {
      int userInput1 = 0;
      int userInput2 = 0;

      cout<<endl<<"Input 1 :"<<endl;
      cin>>userInput1;
      input1->data = userInput1;
      cout<<"Input 2 : "<<endl;
      cin>>userInput2;
      input2->data = userInput2;
      //breadthFirst(); //Note : this need to change into a proper calculator to determine the output
      cout<<"Output : "<< output1->data;
    }
  }

  void adjustWeights(float learnRate, float momentumRate)
  {
    for(list layer : layers)
    {
      for(Node *layerNode : layer)
      {
        for(Link *link : layerNode->nexts)
        { 
          link->props->momentum_multiplier = learnRate * link->props->gradient + momentumRate * link->props->momentum_multiplier;
          //Batch learning : use the same formula as above but replace link->props->gradient with the total of all gradients

          link->props->weight = link->props->weight + link->props->momentum_multiplier;
        }
      }
    }
  }

  void display()
  {
    for (list nodeList : layers)
    {
      cout << endl;
      for (Node *node : nodeList)
      {
        cout << node->data;
      }
      cout << endl;
      for (Node *node : nodeList)
      {
        cout << "/";
        for (Link *link : node->nexts)
        {
          cout << link->props->weight;
          link->props->weight = link->props->weight + 0.8;
        }
      }
    }
  }

  void clearGradients()
  {
    for (list nodeList : layers)
    {

      for (Node *node : nodeList)
      {
        for (Link *link : node->nexts)
        {
          link->props->gradient = 0;
        }
      }
    }
  }

  void depthFirst()
  {

    list<list<Node *>>::reverse_iterator it;
    for (it = layers.rbegin(); it != layers.rend(); it++)
    {
      list<Node *> layerNodes = *it;

      // calculation for all hidden layers
      // δ[i] = σ'( Σ (O[j]) ) * Σ ( W[i] * δ[k] )
      for (Node *layerNode : layerNodes)
      {
        // calculation for OUTPUT nodes only
        if (layerNode->nexts.empty())
        {
          //cout << endl << layerNode->data;
          float error = layerNode->data - layerNode->ideal;
          float delta = (-1) * error * layerNode->dfSum;
          layerNode->delta = delta;
        }       
        
        // skip calculations for BIAS and INPUT nodes
       // if (!layerNode->prevs.empty())
       if(!(layerNode->nodeType == BIAS) && !(layerNode->nodeType == OUTPUT))
        {
          float sumOfWeight = 0;
          for (Link *link : layerNode->nexts)
          {
            sumOfWeight = sumOfWeight + link->props->weight;
          }

          for (Link *link : layerNode->nexts)
          {

            layerNode->delta = layerNode->dfSum * sumOfWeight * link->node->delta;

            //online training
            link->props->gradient = layerNode->data * link->node->delta;
            
            //batch training | we use the total of the gradient
            //link->props->gradient = link->props->gradient + (layerNode->data * link->node->delta);
          }
        }
      }
    }
  }

  void breadthFirst()
  {
    //skip first layer INPUT nodes
    int layerLevel = 1;
    for (list layer : layers)
    {
      if (layerLevel > 1)
      {
        for (Node *layerNode : layer)
        {
          float sum = 0;
          for (Link *link : layerNode->prevs)
          {
            sum = sum + (link->props->weight) * (link->node->data);
          }
          // cout << endl << "sum : " << sum;
          layerNode->sum = sum;
          layerNode->data = sigmoid(sum);
          layerNode->dfSum = sigmoidDerivative(sum); //NOTE:or sigmoid value as input
        }
      }
      layerLevel++;
    }
  }

  float sigmoid(float x)
  {
    // faster as mentioned here
    // https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
    // return value / (1 + abs(value));
    // f=1/(1+e^(-x))
    const float e = 2.71828;
    return 1 / (1 + pow(e, x));
  }

  float sigmoidDerivative(float x)
  {
    // f'=(e^x)/[e^(2x)+2e^(x)+1]
    const float e = 2.71828;
    return pow(e, x) / (pow(e, 2 * x) + 2 * pow(e, x) + 1);
  }

  float error()
  {
    // [(I1-A1)^2 + (I2-A2)^2 + (I3-A3)^2 + (In-An)^2]/ n

    float error = 0.0;
    float sum = 0.0;
    int nodeCount = 1;

    //do error calculation on the last OUTPUT nodes
    for (Node *node : layers.back())
    {
      sum = sum + pow((node->ideal - node->data), 2);
      nodeCount++;
    }
    return sum / nodeCount;
  }
};

int main()
{
  NeuralNetwork nn;
  nn.builder();

  return 0;
}
