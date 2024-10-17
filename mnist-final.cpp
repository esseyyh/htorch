#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>


#include <htorch.h>
#include <iostream>
#include <stdexcept>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>


//reference for the following code 
//
//https://github.com/HyTruongSon/Neural-Network-MNIST-CPP/tree/master
using namespace std;





// opening and saving files for mnist data base 

const string training_image_fn = "mnist/train-images.idx3-ubyte";
const string training_label_fn = "mnist/train-labels.idx1-ubyte";
const string model_fn = "model-neural-network.dat";

const string testing_image_fn = "mnist/t10k-images.idx3-ubyte";
const string testing_label_fn = "mnist/t10k-labels.idx1-ubyte";



const string report_fn = "training-report.dat";
// Number of training images to use 
const int nTraining = 60000;
const int nTesting = 10000;


// Image size in MNIST database 
const int width = 28;
const int height = 28;

// n1 = Number of input neurons
// n2 = Number of hidden neurons
// n3 = Number of output neurons
// epochs = Number of iterations for back-propagation algorithm
// learning_rate = Learing rate
// momentum = Momentum (heuristics to optimize back-propagation algorithm)
// epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 128; 
const int n3 = 10; 
const int epochs = 1;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;


//
//// Initialize the tensors using dynamic list of dimensions (variadic arguments)
//htorch::Tensor *w1 = new htorch::Tensor(n1, n2);    // w1 is a tensor of size n1 x n2
//htorch::Tensor *delta1 = new htorch::Tensor(n1, n2); // delta1 is also n1 x n2
//htorch::Tensor *out1 = new htorch::Tensor(n1);      // out1 is a tensor of size n1 (1D)
//
//
//
//htorch::Tensor *w2 = new htorch::Tensor(n2, n3);    // w1 is a tensor of size n1 x n2
//htorch::Tensor *delta2 = new htorch::Tensor(n2, n3); // delta1 is also n1 x n2
//htorch::Tensor *out2 = new htorch::Tensor(n1);      // out1 is a tensor of size n1 (1D)
//
//
//




htorch::Tensor *w1 = new htorch::Tensor(n1, n2);     // Weights from input to hidden layer
htorch::Tensor *delta1 = new htorch::Tensor(n1, n2); // Gradient of the weights
htorch::Tensor *out1 = new htorch::Tensor(1,n1);       // Output from layer 1 (input)

// From Hidden layer (n2 neurons) to Output layer (n3 neurons)
htorch::Tensor *w2 = new htorch::Tensor(n2, n3);     // Weights from hidden to output layer
htorch::Tensor *delta2 = new htorch::Tensor(n2, n3); // Gradient of the weights
htorch::Tensor *in2 = new htorch::Tensor(1,n2);        // Input to layer 2 (hidden)
htorch::Tensor *out2 = new htorch::Tensor(1,n2);       // Output from hidden layer
htorch::Tensor *theta2 = new htorch::Tensor(1,n2);     // Bias/activation for hidden layer

// Output layer
htorch::Tensor *in3 = new htorch::Tensor(1,n3);        // Input to the output layer
htorch::Tensor *out3 = new htorch::Tensor(1,n3);       // Output from the output layer
htorch::Tensor *theta3 = new htorch::Tensor(1,n3);     // Bias/activation for output layer
htorch::Tensor *expected = new htorch::Tensor(1,n3);   // Expected output (labels) for training

// Image (MNIST: 28x28 grayscale images)
htorch::Tensor *d = new htorch::Tensor(width, height); // Image tensor (28x28)





ifstream image;
ifstream label;
ofstream report;

void about() {
	// Details
	cout << "**************************************************" << endl;
	cout << "*** Training Neural Network for MNIST database ***" << endl;
	cout << "**************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "No. iterations: " << epochs << endl;
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
	cout << endl;
	cout << "Training image data: " << training_image_fn << endl;
	cout << "Training label data: " << training_label_fn << endl;
	cout << "No. training sample: " << nTraining << endl << endl;
}


void about_test() {
	// Details
	cout << "*************************************************" << endl;
	cout << "*** Testing Neural Network for MNIST database ***" << endl;
	cout << "*************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "Testing image data: " << testing_image_fn << endl;
	cout << "Testing label data: " << testing_label_fn << endl;
	cout << "No. testing sample: " << nTesting << endl << endl;
}
void init_tensors() {
    // Initialize weights and deltas for Layer 1 to Layer 2 (Input to Hidden)
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; ++j) {
            int sign = rand() % 2;
            double value = (double)(rand() % 6) / 10.0;  // Random value between 0 and 0.5
            if (sign == 1) {
                value = -value;  // Randomize sign
            }
            w1->value(i, j) = value;   // Assign value to w1 tensor
            //delta1->value(i, j) = 0.0; // Initialize delta1 with 0
        }
    }
    
    // Initialize output tensor for layer 1 (just setting to 0 here, modify as needed)
    for (int i = 0; i < n1; i++) {
        out1->value(i) = 0.0;  // Initialize out1 tensor with 0
    }

    // Initialize weights and deltas for Layer 2 to Layer 3 (Hidden to Output)
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n3; ++j) {
            int sign = rand() % 2;
            double value = (double)(rand() % 10 + 1) / (10.0 * n3);  // Random value between 0 and 0.1
            if (sign == 1) {
                value = -value;  // Randomize sign
            }
            w2->value(i, j) = value;   // Assign value to w2 tensor
            //delta2->value(i, j) = 0.0; // Initialize delta2 with 0
        }
    }

    // Initialize the in2, out2, and theta2 tensors
    for (int i = 0; i < n2; i++) {
        in2->value(0,i) = 0.0;      // Initialize in2 with 0
        out2->value(0,i) = 0.0;     // Initialize out2 with 0
        theta2->value(0,i) = 0.0;   // Initialize theta2 with 0 (adjust if bias initialization is needed)
    }

    // Initialize the in3, out3, and theta3 tensors for the output layer
    for (int i = 0; i < n3; i++) {
        in3->value(0,i) = 0.0;      // Initialize in3 with 0
        out3->value(0,i) = 0.0;     // Initialize out3 with 0
        theta3->value(0,i) = 0.0;   // Initialize theta3 with 0 (adjust if bias initialization is needed)
    }
    
    // You can initialize the expected output tensor similarly if needed:
    for (int i = 0; i < n3;i++) {
        expected->value(0,i) = 0.0; // Initialize expected with 0 or a label if needed
    }
}
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
void perceptron() {
    // Initialize in2 to 0 for the hidden layer inputs
    for (int i = 0; i < n2; i++) {
        in2->value(0,i) = 0.0;
    }

    // Initialize in3 to 0 for the output layer inputs
    for (int i = 0; i < n3; i++) {
        in3->value(0,i) = 0.0;
    }

    // Propagate inputs from Layer 1 (Input Layer) to Layer 2 (Hidden Layer)
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; ++j) {
            in2->value(0,j) += out1->value(0,i) * w1->value(i, j);  // Multiply out1 and w1 tensors
        }
    }

    // Apply the activation function (sigmoid) to hidden layer outputs
    for (int i = 0; i < n2; i++) {
        out2->value(0,i) = sigmoid(in2->value(0,i));  // Apply sigmoid to in2 tensor
    }

    // Propagate inputs from Layer 2 (Hidden Layer) to Layer 3 (Output Layer)
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n3; ++j) {
            in3->value(0,j) += out2->value(0,i) * w2->value(i, j);  // Multiply out2 and w2 tensors
        }
    }


    // Apply the activation function (sigmoid) to output layer outputs
    for (int i = 0; i < n3; i++) {
        out3->value(0,i) = sigmoid(in3->value(0,i));  // Apply sigmoid to in3 tensor
    }
    //cout<<out3->value.shape()[1]<<endl;

   


    //int guessed_label = std::distance(out3->data(), std::max_element(out3->data(), out3->data() + n3));
    //cout << "Guessed label: " << out3->value << endl;
}

double square_error() {
    double res = 0.0;

    // Loop through the output layer neurons
    for (int i = 0; i < n3; ++i) {  // Adjusted loop to start from 0
        double diff = out3->value(0,i) - expected->value(0,i);  // Access tensor values using the values method
        res += diff * diff;  // Accumulate squared error
    }

    res *= 0.5;  // Multiply by 0.5
    return res;  // Return the computed squared error
}













int input_test() {
	// Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d->value(i,j) = 0; 
			} else {
				d->value(i,j) = 1;
			}
        }
	}

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1->value(pos) = d->value(i,j);
        }
	}

	// Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
		expected->value(0,i) = 0.0;
	}
    expected->value(0,number + 1) = 1.0;
        
    return (int)(number);
}






















void back_propagation() {
    double sum;

    // Calculate the gradients for the output layer (Layer 3)
    for (int i = 0; i < n3; ++i) {  // Adjusted to 0-based indexing
        theta3->value(0,i) = out3->value(0,i) * (1 - out3->value(0,i)) * (expected->value(0,i) - out3->value(0,i));
    }

    // Calculate the gradients for the hidden layer (Layer 2)
    for (int i = 0; i < n2; ++i) {  // Adjusted to 0-based indexing
        sum = 0.0;
        for (int j = 0; j < n3; ++j) {  // Adjusted to 0-based indexing
            sum += w2->value(i, j) * theta3->value(0,j);  // Accessing weight tensor
        }
        theta2->value(0,i) = out2->value(0,i) * (1 - out2->value(0,i)) * sum;
    }

    // Update weights for Layer 2 (Hidden layer to Output layer)
    for (int i = 0; i < n2; ++i) {  // Adjusted to 0-based indexing
        for (int j = 0; j < n3; ++j) {  // Adjusted to 0-based indexing
            delta2->value(i, j) = (learning_rate * theta3->value(0,j) * out2->value(0,i)) + (momentum * delta2->value(i, j));
            w2->value(i, j) += delta2->value(i, j);  // Update weights
        }
    }

    // Update weights for Layer 1 (Input layer to Hidden layer)
    for (int i = 0; i < n1; ++i) {  // Adjusted to 0-based indexing
        for (int j = 0; j < n2; ++j) {  // Adjusted to 0-based indexing
            delta1->value(i, j) = (learning_rate * theta2->value(0,j) * out1->value(0,i)) + (momentum * delta1->value(i, j));
            w1->value(i, j) += delta1->value(i, j);  // Update weights
        }
    }
    //cout<<"-------------------------------------------------------------------------------------------------"<<endl;
    //cout<<delta2->value<<endl;

    //cout<<"-------------------------------------------------------------------------------------------------"<<endl;
}





int learning_process() {
    // Initialize delta tensors to zero
    for (int i = 0; i < n1; ++i) {  // Adjusted to 0-based indexing
        for (int j = 0; j < n2; ++j) {  // Adjusted to 0-based indexing
            delta1->value(i, j) = 0.0;
        }
    }


    for (int i = 0; i < n2; ++i) {  // Adjusted to 0-based indexing
        for (int j = 0; j < n3; ++j) {  // Adjusted to 0-based indexing
            delta2->value(i, j) = 0.0;
        }
    }

    // Training loop for the specified number of epochs
    for (int i = 0; i < epochs; ++i) {  // Adjusted to 0-based indexing
        perceptron();          // Forward pass
        back_propagation();    // Backward pass
        if (square_error() < epsilon) {  // Check for convergence
            return i;  // Return the number of epochs completed
        }
    }
    return epochs;  // Return total epochs if not converged
}


void input() {
    // Reading image
    char number;
    
    // Create a tensor for the image with dimensions (height, width)
    //htorch::Tensor d(height, width); // Assuming height and width are already defined

    for (int j = 0; j < height; ++j) {  // Adjusted to 0-based indexing
        for (int i = 0; i < width; ++i) {  // Adjusted to 0-based indexing
            image.read(&number, sizeof(char));
            // Set tensor values based on pixel data
            d->value(j, i) = (number == 0) ? 0 : 1; 
        }
    }
    
    cout << "Image:" << endl;
    for (int j = 0; j < height; ++j) {  // Adjusted to 0-based indexing
        for (int i = 0; i < width; ++i) {  // Adjusted to 0-based indexing
            cout << d->value(j, i);  // Access tensor values
        }
        cout << endl;
    }

    // Flatten the tensor to out1
    for (int j = 0; j < height; ++j) {  // Adjusted to 0-based indexing
        for (int i = 0; i < width; ++i) {  // Adjusted to 0-based indexing
            int pos = i + j * width;  // Flattening index calculation
            out1->value(0,pos) = d->value(j, i);  // Access tensor values
        }
    }

    // Reading label
    label.read(&number, sizeof(char));
    
    // Initialize expected tensor
    //htorch::Tensor expected(n3, 1);  // Create a tensor for expected output
    for (int i = 0; i < n3; ++i) {  // Adjusted to 0-based indexing
        expected->value(0,i) = 0.0;  // Initialize expected values to 0.0
    }
    expected->value(0,static_cast<int>(number)) = 1.0;  // Set the correct class label

    cout << "Label: " << static_cast<int>(number) << endl;
}
void write_matrix(const string& file_name) {
    ofstream file(file_name.c_str(), ios::out);

    // Input layer - Hidden layer
    for (int i = 0; i < n1; ++i) {  // Using zero-based indexing
        for (int j = 0; j < n2; ++j) {
            file << w1->value(i, j) << " ";  // Accessing values using the tensor method
        }
        file << endl;
    }

    // Hidden layer - Output layer
    for (int i = 0; i < n2; ++i) {  // Using zero-based indexing
        for (int j = 0; j < n3; ++j) {
            file << w2->value(i, j) << " ";  // Accessing values using the tensor method
        }
        file << endl;
    }

    file.close();
}
int main(int argc, char *argv[]) {
    about();

    report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary); // Binary label file

    // Reading file headers
    char number;
    for (int i = 0; i < 16; ++i) {  // Using zero-based indexing
        image.read(&number, sizeof(char));
    }
    for (int i = 0; i < 8; ++i) {  // Using zero-based indexing
        label.read(&number, sizeof(char));
    }

    // Neural Network Initialization
    init_tensors();

    for (int sample = 0; sample < nTraining; ++sample) {  // Using zero-based indexing
        cout << "Sample " << sample + 1 << endl;

        // Getting (image, label)
        input();


        // Learning process: Perceptron (Forward procedure) - Back propagation
        int nIterations = learning_process();


        // Write down the squared error
        cout << "No. iterations: " << nIterations << endl;
        printf("Error: %0.6lf\n\n", square_error());
        report << "Sample " << sample + 1 << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;

        //cout<<w1->value<<endl;
        //cout<<delta1->value<<endl;
        //cout<<w2->value<<endl;
        //cout<<delta2->value<<endl;
        // Save the current network (weights)
        if ((sample + 1) % 100 == 0) {  // Using one-based for output
            cout << "Saving the network to " << model_fn << " file." << endl;
            write_matrix(model_fn);
        }
    }

    // Save the final network
    write_matrix(model_fn);

    report.close();
    image.close();
    label.close();

    cout<<"----------------------------------------------------------------------------------------------------------------------"<<endl;
    cout<<"----------------------------------------------------------------------------------------------------------------------"<<endl;
    cout<<"----------------------------------------------------------------------------------------------------------------------"<<endl;
    cout<<"testing time baby"<<endl;


    about_test();
	
    report.open(report_fn.c_str(), ios::out);
    image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

	// Reading file headers
    char number_test;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number_test, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number_test, sizeof(char));
	}
		
	// Neural Network Initialization
    //init_array(); // Memory allocation
    //load_model(model_fn); // Load model (weight matrices) of a trained Neural Network
    
    int nCorrect = 0;
    for (int sample = 1; sample <= nTesting; ++sample) {
        cout << "Sample " << sample << endl;
        
        // Getting (image, label)
        int label = input_test();
		
		// Classification - Perceptron procedure
        perceptron();
        
        // Prediction
        int predict = 1;
        for (int i = 2; i <= n3; ++i) {
			if (out3->value(0,i) > out3->value(0,predict)) {
				predict = i;
			}
		}
		--predict;

		// Write down the classification result and the squared error
		double error = square_error();
		printf("Error: %0.6lf\n", error);
		
		if (label == predict) {
			++nCorrect;
			cout << "Classification: YES. Label = " << label << ". Predict = " << predict << endl << endl;
			report << "Sample " << sample << ": YES. Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
		} else {
			cout << "Classification: NO.  Label = " << label << ". Predict = " << predict << endl;
			cout << "Image:" << endl;
			for (int j = 1; j <= height; ++j) {
				for (int i = 1; i <= width; ++i) {
					cout << d->value(i,j);
				}
				cout << endl;
			}
			cout << endl;
			report << "Sample " << sample << ": NO.  Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
		}
    }

	// Summary
    double accuracy = (double)(nCorrect) / nTesting * 100.0;
    cout << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    printf("Accuracy: %0.2lf\n", accuracy);
    
    report << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    report << "Accuracy: " << accuracy << endl;

    report.close();
    image.close();
    label.close();

    return 0;
}

