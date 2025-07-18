module {
  func.func @main(%arg0: tensor<33x26xf32>, %arg1: tensor<82x82x79x26x66xi64>, %arg2: tensor<11x42x73xi1>, %arg3: tensor<11x1x73xi1>) -> (tensor<33x52xf32>, tensor<82x82x79x26x66xi64>, tensor<11x42x73xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<33x26xf32>) -> tensor<33x26xf32>
    %1 = tosa.clamp %0 {min_val = -4.800000e+01 : f32, max_val = 7.500000e+01 : f32} : (tensor<33x26xf32>) -> tensor<33x26xf32>
    %2 = tosa.rsqrt %1 : (tensor<33x26xf32>) -> tensor<33x26xf32>
    %3 = tosa.clz %arg1 : (tensor<82x82x79x26x66xi64>) -> tensor<82x82x79x26x66xi64>
    %4 = tosa.concat %2, %0 {axis = 1 : i32} : (tensor<33x26xf32>, tensor<33x26xf32>) -> tensor<33x52xf32>
    %5 = tosa.arithmetic_right_shift %3, %3 {round = false} : (tensor<82x82x79x26x66xi64>, tensor<82x82x79x26x66xi64>) -> tensor<82x82x79x26x66xi64>
    %6 = tosa.logical_xor %arg2, %arg3 : (tensor<11x42x73xi1>, tensor<11x1x73xi1>) -> tensor<11x42x73xi1>
    return %4, %5, %6 : tensor<33x52xf32>, tensor<82x82x79x26x66xi64>, tensor<11x42x73xi1>
  }
}
