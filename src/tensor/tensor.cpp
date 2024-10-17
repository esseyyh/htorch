#include "../../include/tensor/tensor.h"
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

Tensor::Tensor(){};
Tensor::Tensor(float v) {
  value.reshape({1});
  value[0] = v;
  grad.reshape({1});
  grad[0] = 0.0;
};

Tensor::Tensor(int v) {
  value.reshape({1});
  value[0] = float(v);
  grad.reshape({1});
  grad[0] = 0.0;
};

void Tensor::fill(float val) { value.fill(val); }
void Tensor::print() {
  std::cout << "Value: " << value << std::endl;
  std::cout << "Grad: " << grad << std::endl;
}

} // namespace htorch
