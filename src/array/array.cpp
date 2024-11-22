#include "../../include/array/array.h"
#include <iostream>
#include <stdexcept>
//#include <vector>





//class Array {
//public:
//    float* data;   // Pointer to raw data
//    int size;      // Number of dimensions
//    int* shape;    // Array shape
//    int* strides; 
//    int linear_indices;// Strides for each dimension
//
//    // Constructor accepts the dims and sets them to zero, calulates the size,shape and stride as well
//
//
//template<typename... Dims>
//Array(Dims... dimensions) {
//    constexpr int num_dims = sizeof...(Dims);                                   // input dims
//    
//    size = num_dims;                                                           // size of the array             
//    shape = new int[size];                                                     // define the length of the shape array
//    
//
//
//    std::vector<int> shape_vec = {static_cast<int>(dimensions)...};             // shape in vector to calulate the total size
//    
//    // Calculate total size
//    int total_size = 1;
//    for (int dim : shape_vec) {
//        total_size *= dim;
//    }
//    linear_indices=total_size;
//    
//    // Allocate memory for shape, strides, and data
//
//
//    std::copy(shape_vec.begin(), shape_vec.end(), shape);
//    
//    // Compute strides
//    strides = new int[size];
//    strides[size-1] = 1;
//    for (int i = size-2; i >= 0; --i) {
//        strides[i] = strides[i+1] * shape[i+1];
//    }
//    
//    // Allocate linear array for data, initialize to zero
//    data = new float[total_size]();
//}
//
//
//
//  //Array(float* data, int ndim, int* shape, int* strides)
//  //      : data(data), ndim(ndim), shape(shape), strides(strides) {}
//
//
//    // General template for N-dimensional indexing
//    template <typename... Indices>
//    float& operator()(Indices... indices) {
//        return get_offset(indices...);
//    }
//
//    // Helper function to compute offset with multiple indices
//    template <typename... Indices>
//    float& get_offset(Indices... indices) {
//        int offset = 0;
//        int dims[] = {indices...};  // Unpack indices into array
//        for (int dim = 0; dim < sizeof...(indices); ++dim) {
//            if (dims[dim] < 0 || dims[dim] >= shape[dim]) {
//                throw std::out_of_range("Index out of bounds.");
//            }
//            offset += dims[dim] * strides[dim];
//        }
//        return data[offset];
//    }
//
    void Array::print() {
        std::cout << "ndim: " << size << "\n";
        std::cout << "shape: ";
        for (int i = 0; i < size; ++i) {
            std::cout << shape[i] << " ";
        }
        std::cout << "\n";
        std::cout << "strides: ";
        for (int i = 0; i < size; ++i) {
            std::cout << strides[i] << " ";
        }
        std::cout << "\n";
    }
//};



float _getrandomFloat(float min, float max)
{
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

//
//
//
//template <typename... Indices>
//Array zeros(Indices... indices){
//return Array(indices...);
//}
//
//
//template <typename... Indices>
//Array random(Indices... indices){
//Array temp =Array(indices...);
//    for(int i=0;i<temp.linear_indices;i++){
//        temp.data[i]=_getrandomFloat(0,1);
//    }
//    return temp;
//}
//
//
//template <typename... Indices>
//Array ones(Indices... indices){
//Array temp =Array(indices...);
//    for(int i=0;i<temp.linear_indices;i++){
//        temp.data[i]=1.0;
//    }
//    return temp;
//}
//

Array linspace(float start, float end, int num){
    Array temp =Array(num);
    float step = (end-start)/(num-1);
    for(int i=0;i<num;i++){
        temp.data[i]=start+i*step;
    }
    return temp;
}

//
//
//
//template <typename... Indices>
//Array arrange(Indices... indices){
//
//    constexpr int num_dims = sizeof...(Indices);
//
//    std::vector<float> shape_vec = {static_cast<float>(indices)...};             // shape in vector to calulate the total size
//    
//
//    if (num_dims==1){
//        Array temp =Array(indices...);
//        for(int i=0;i<shape_vec[0];i++){
//            temp.data[i]=i;
//            return temp;
//        
//        }
//    }
//    else if (num_dims == 2){
//        int size=(shape_vec[1]-shape_vec[0]);
//        Array temp=Array(size);
//        for(int i=0;i<size;i++){
//            temp.data[i]=shape_vec[0]+i;
//        }
//        return temp;
//    }
//
//    else if (num_dims == 3){
//        int size=(shape_vec[1] - shape_vec[0]) / shape_vec[2];
//        Array temp=Array(size);
//        for(int i=0;i<size;i++){
//            temp.data[i]=shape_vec[0] + i*shape_vec[2];
//        }
//
//
//        return temp;
//    }
//    else{
//        throw std::out_of_range("Index out of bounds.");
//
//    }
//    return 0;
//}
//
//
Array eye(int size){
Array temp = Array(size,size);
    for (int i = 0; i < size; i++){
        temp(i,i)=1;
    }
return temp;

}



Array fill(Array &arr, float value){

    for(int i=0;i<arr.linear_indices;i++){
        arr.data[i]=value;
    }
    return arr;
}


void matmul( Array &a, Array &b, Array &C) {
    // Check for valid dimensions
    if (a.shape[1] != b.shape[0]) {
        throw std::out_of_range("Mismatched dimensions: incompatible for matrix multiplication.");
    }

    // Ensure 2D matrices
    if (a.size != 2 || b.size != 2 || C.size != 2) {
        throw std::out_of_range("Only 2D matrix multiplication is currenlty supported.");
    }

    // Ensure the output matrix has the correct shape
    if (C.shape[0] != a.shape[0] || C.shape[1] != b.shape[1]) {
        throw std::out_of_range("Output matrix has incorrect dimensions.");
    }

    // Perform the matrix multiplication
    for (int i = 0; i < a.shape[0]; i++) {
        for (int j = 0; j < b.shape[1]; j++) {
            float value = 0.0f; // Initialize the result of the dot product
            for (int k = 0; k < a.shape[1]; k++) { // Shared dimension
                value += a(i, k) * b(k, j); // Accumulate dot product
            }
            C(i, j) = value; // Assign the result
        }
    }
    
}

void add(Array &a, Array &b, Array &C){

    if (a.size != b.size || a.size != C.size) {
        throw std::out_of_range("Mismatched size: incompatible for matrix addition.");
    }

    for(int i=0;i<a.size;i++){
        if(a.shape[i]!=b.shape[i] || a.shape[i]!=C.shape[i]){
            throw std::out_of_range("Mismatch in dimensions : incompatible for matrix addition.");
        }
    }

    for(int i=0;i<a.linear_indices;i++){
        C.data[i]=a.data[i]+b.data[i];
    }


} 
void multiply(Array &a, Array &b, Array &C){

    if (a.size != b.size || a.size != C.size) {
        throw std::out_of_range("Mismatched size: incompatible for matrix multiplication.");
    }

    for(int i=0;i<a.size;i++){
        if(a.shape[i]!=b.shape[i] || a.shape[i]!=C.shape[i]){
            throw std::out_of_range("Mismatch in dimensions : incompatible for matrix multiplication.");
        }
    }

    for(int i=0;i<a.linear_indices;i++){
        C.data[i]=a.data[i]*b.data[i];
    }


}




int main (){



Array a = ones(3,4);
Array b = ones(4,3);
Array c = zeros(3,3);

//matmul(a,b,c);

//
//Array a= eye(7);
//a.print();
//
for(int i =0;i<3;i++){
    for(int j=0;j<3;j++){
        std::cout<<c(i,j)<<" ";
    }
    std::cout<<std::endl;
}

Array a1 = ones(3,3);
Array b1 = ones(3,3);
Array c1 = zeros(3,3);
    multiply(a1,b1,c1);


//
//Array a= eye(7);
//a.print();
//
for(int i =0;i<3;i++){
    for(int j=0;j<3;j++){
        std::cout<<c1(i,j)<<" ";
    }
    std::cout<<std::endl;
}
    return 0;



}


