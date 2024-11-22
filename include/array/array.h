#ifndef HTORCH_ARRAY_H
#define HTORCH_ARRAY_H


#include <iostream>
#include <stdexcept>
#include <vector>





class Array {
public:
    float* data;   // Pointer to raw data
    int size;      // Number of dimensions
    int* shape;    // Array shape
    int* strides; 
    int linear_indices;// Strides for each dimension

    // Constructor accepts the dims and sets them to zero, calulates the size,shape and stride as well


template<typename... Dims>
Array(Dims... dimensions) {
    constexpr int num_dims = sizeof...(Dims);                                   // input dims
    
    size = num_dims;                                                           // size of the array             
    shape = new int[size];                                                     // define the length of the shape array
    


    std::vector<int> shape_vec = {static_cast<int>(dimensions)...};             // shape in vector to calulate the total size
    
    // Calculate total size
    int total_size = 1;
    for (int dim : shape_vec) {
        total_size *= dim;
    }
    linear_indices=total_size;
    
    // Allocate memory for shape, strides, and data


    std::copy(shape_vec.begin(), shape_vec.end(), shape);
    
    // Compute strides
    strides = new int[size];
    strides[size-1] = 1;
    



    for (int i = size-2; i >= 0; --i) {
        strides[i] = strides[i+1] * shape[i+1];
    }
    
    // Allocate linear array for data, initialize to zero
    data = new float[total_size]();
}


    // General template for N-dimensional indexing
    template <typename... Indices>
    float& operator()(Indices... indices) {
        return get_offset(indices...);
    }

    // Helper function to compute offset with multiple indices
    template <typename... Indices>
    float& get_offset(Indices... indices) {
        int offset = 0;
        int dims[] = {indices...};  // Unpack indices into array
        for (int dim = 0; dim < sizeof...(indices); ++dim) {
            if (dims[dim] < 0 || dims[dim] >= shape[dim]) {
                throw std::out_of_range("Index out of bounds.");
            }
            offset += dims[dim] * strides[dim];
        }
        return data[offset];
    }

    void print();
};



float _getrandomFloat(float min, float max);



Array fill(Array &arr, float value);

template <typename... Indices>
Array zeros(Indices... indices){
return Array(indices...);
}


template <typename... Indices>
Array random(Indices... indices){
Array temp =Array(indices...);
    for(int i=0;i<temp.linear_indices;i++){
        temp.data[i]=_getrandomFloat(0,1);
    }
    return temp;
}


template <typename... Indices>
Array ones(Indices... indices){
Array temp =Array(indices...);
    for(int i=0;i<temp.linear_indices;i++){
        temp.data[i]=1.0;
    }
    return temp;
}


Array linspace(float start, float end, int num);




template <typename... Indices>
Array arrange(Indices... indices){

    constexpr int num_dims = sizeof...(Indices);

    std::vector<float> shape_vec = {static_cast<float>(indices)...};             // shape in vector to calulate the total size
    

    if (num_dims==1){
        Array temp =Array(indices...);
        for(int i=0;i<shape_vec[0];i++){
            temp.data[i]=i;
            return temp;
        
        }
    }
    else if (num_dims == 2){
        int size=(shape_vec[1]-shape_vec[0]);
        Array temp=Array(size);
        for(int i=0;i<size;i++){
            temp.data[i]=shape_vec[0]+i;
        }
        return temp;
    }

    else if (num_dims == 3){
        int size=(shape_vec[1] - shape_vec[0]) / shape_vec[2];
        Array temp=Array(size);
        for(int i=0;i<size;i++){
            temp.data[i]=shape_vec[0] + i*shape_vec[2];
        }


        return temp;
    }
    else{
        throw std::out_of_range("Index out of bounds.");

    }
    return 0;
}


Array eye(int size);



void matmul( Array &a, Array &b, Array &C);
void add(Array &a, Array &b, Array &C);
void multiply(Array &a, Array &b, Array &C);






#endif // HTORCH_ARRAY_H
