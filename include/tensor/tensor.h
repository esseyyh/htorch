
#ifndef HTORCH_TENSOR_H
#define HTORCH_TENSOR_H

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

class Tensor {
private:
public:
  xt::xarray<double> value; // value of the tensor
  xt::xarray<double> grad;  // grad of the tensor
  std::function<void()> backward_fn;
  std::vector<std::shared_ptr<Tensor>> parents;

  void init_grad();

  // constructors for creation using different datatypes
  Tensor();
  Tensor(const std::string &v);
  Tensor(float v);
  Tensor(int v); //

  template <typename... Args> Tensor(Args... args) {
    std::vector<int> shape = {static_cast<int>(args)...};
    value.resize(shape);
    grad.resize(shape);
    grad.fill(0.0);
  }

  template <typename T, std::size_t N, typename A, bool Init>
  Tensor(const xt::svector<T, N, A, Init> &shape) {
    std::vector<size_t> size_t_shape(shape.begin(), shape.end());
    value.resize(size_t_shape);
    grad.resize(size_t_shape);
  }

  void fill(float val);
  void print();
  void backward() {

    if (grad.size() == 0) {
      grad = xt::ones_like(value);
    }
    if (backward_fn) {
      backward_fn();
    }

    if (!parents.empty()) {
      for (auto &parent : parents) {

        parent->backward();
      }
    }
  }

  void zero_grad() {
    grad = xt::zeros_like(value);
    for (auto &parent : parents) {
      parent->zero_grad();
    }
  }
};

} // namespace htorch

#endif // TENSOR_H
