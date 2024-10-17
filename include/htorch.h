
#ifndef HTORCH_HTORCH_H
#define HTORCH_HTORCH_H

#include "xtensor/xadapt.hpp"
#include <any>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xtensor.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>
#include <xtensor/xarray.hpp>


#include "tensor/tensor.h"


namespace htorch{
template <typename... Args> Tensor zeros(Args... args) {
  Tensor val(std::forward<Args>(args)...);
  val.fill(0.0);
  return val;
}
template <typename... Args> Tensor ones(Args... args) {
  Tensor val(std::forward<Args>(args)...);
  val.fill(1.0);
  return val;
}
template <typename... Args> Tensor full(Args... args, int value) {
  Tensor val(std::forward<Args>(args)...);
  val.fill(value);
  return val;
}
Tensor zeros_like(const Tensor &obj);
Tensor ones_like(const Tensor &obj); 

Tensor full_like(const Tensor &obj, float valf);








 
template <typename T>
struct InnermostType {
    using type = T;  // By default, the innermost type is the type itself
};

// Specialization for std::vector
template <typename T, typename A>
struct InnermostType<std::vector<T, A>> {
    using type = typename InnermostType<T>::type;
};

// Specialization for std::array
template <typename T, std::size_t N>
struct InnermostType<std::array<T, N>> {
    using type = typename InnermostType<T>::type;
};


// Function to determine the type of the innermost elements  will be called in the fromvector to ensure type int and float
template <typename T>
bool checkInnermostType(T) {
    using ElementType = typename InnermostType<T>::type;

    if constexpr (std::is_same<ElementType, int>::value) {
        return true;
    } else if constexpr (std::is_same<ElementType, float>::value) {
        return true;
    } else if constexpr (std::is_same<ElementType, double>::value) {
        return true;
    } else if constexpr (std::is_same<ElementType, std::string>::value) {
        return false;
    } else {
        return false;
    }
};



// a function to check if the nested vector is jagged since tensor should not be created by jagged vectors 

////// Helper function to check if the type is a vect
template <typename T> struct is_vector {
static constexpr bool value = false;
};
//
template <typename T> struct is_vector<std::vector<T>> {
  static constexpr bool value = true;
};
//
template <typename T> bool isntJagged(const T &vec) {
  return false;
}
template <typename T> bool isntJagged(const std::vector<T> &vec) {
  if (vec.empty())
    return false;

  if constexpr (is_vector<T>::value) {
    size_t first_size = vec[0].size();
    for (const auto &sub_vec : vec) {
      if (sub_vec.size() != first_size || isntJagged(sub_vec)) {
        return false;
      }
    }
  }

  return true;
};
//
//
//
//
//
//// flatten a nested array since xtensor only accepts a flattned vector to make an xarray from 

template<typename T>
std::pair<std::vector<T>, std::vector<size_t>> flatten_dynamic_vector(const std::any& nested_vec) {
    std::vector<T> flattened;
    std::vector<size_t> shape;

    std::function<void(const std::any&, size_t)> flatten = [&](const std::any& item, size_t depth) {
        if (item.type() == typeid(T)) {
            flattened.push_back(std::any_cast<T>(item));
        } else if (item.type() == typeid(std::vector<std::any>)) {
            const auto& vec = std::any_cast<const std::vector<std::any>&>(item);
            if (shape.size() <= depth) {
                shape.push_back(vec.size());
            }
            for (const auto& subitem : vec) {
                flatten(subitem, depth + 1);
            }
        } else if (item.type() == typeid(std::vector<T>)) {
            const auto& vec = std::any_cast<const std::vector<T>&>(item);
            if (shape.size() <= depth) {
                shape.push_back(vec.size());
            }
            for (const auto& subitem : vec) {
                flattened.push_back(subitem);
            }
        } else {
            throw std::runtime_error("Unsupported type in nested vector");
        }
    };

    flatten(nested_vec, 0);
    return {flattened, shape};
}

template<typename T>
std::any to_any_vector(const T& vec) {
    if constexpr (std::is_arithmetic_v<T>) {
        return vec;
    } else if constexpr (std::is_same_v<T, std::vector<typename T::value_type>>) {
        std::vector<std::any> result;
        for (const auto& item : vec) {
            result.push_back(to_any_vector(item));
        }
        return result;
    } else {
        throw std::runtime_error("Unsupported type in nested vector");
    }
}


template <typename T> 
Tensor from_vector( std::vector<T> vec) {


    auto [flattened_float, shape_float] = flatten_dynamic_vector<float>(to_any_vector(vec));

    auto adapted = xt::adapt(flattened_float, shape_float);

    xt::xarray<double> bun=adapted;
    htorch::Tensor float_adapted;
    float_adapted.value=bun;
    float_adapted.grad = xt::zeros_like(float_adapted.value);
    return float_adapted ;
    }





Tensor multiply( Tensor& a, Tensor& b);
Tensor add(Tensor& a,  Tensor& b);
}
#endif
