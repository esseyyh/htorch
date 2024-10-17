#include "../include/htorch.h"
#include <iostream>

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

namespace htorch {


Tensor zeros_like(const htorch::Tensor &obj) {


Tensor val;
  val.value.resize(obj.value.shape());
  val.value.fill(0.0);
  return val;
}
htorch::Tensor ones_like(const htorch::Tensor &obj) {
  htorch::Tensor val;
  val.value.resize(obj.value.shape());
  val.value.fill(1.0);
  return val;
}

htorch::Tensor full_like(const htorch::Tensor &obj, float valf) {
  htorch::Tensor val;
  val.value.resize(obj.value.shape());
  val.value.fill(valf);
  return val;
}






Tensor multiply( Tensor& a, Tensor& b)
{
    Tensor result(a.value.shape());  // Create empty tensor with the same shape
    result.value = a.value * b.value;
    auto a_ptr = std::make_shared<Tensor>(a);
    auto b_ptr = std::make_shared<Tensor>(b);
    result.parents = {a_ptr, b_ptr};
    result.backward_fn = [&a, &b, &result]() {
        a.grad += result.grad * b.value;
        b.grad += result.grad * a.value;
    };
    return result;
}





Tensor add(Tensor& a,  Tensor& b) {
    Tensor result(a.value.shape());
    //result.grad.resize(a.value.shape());// Create empty tensor with the same shape
    result.value = a.value + b.value;
    auto a_ptr = std::make_shared<Tensor>(a);
    auto b_ptr = std::make_shared<Tensor>(b);
    result.parents = {a_ptr, b_ptr};
    result.backward_fn = [&a,&b, &result]() {
       

        a.grad += result.grad;
        b.grad += result.grad;
    };
    return result;
}
   

}
