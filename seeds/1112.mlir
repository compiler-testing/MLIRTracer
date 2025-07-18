module {
  func.func @main(%arg0: tensor<98xi32>, %arg1: tensor<1xi32>, %arg2: tensor<32xf32>) -> (tensor<98xi32>, tensor<32xf32>, tensor<1xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<98xi32>, tensor<1xi32>) -> tensor<98xi32>
    %1 = tosa.log %arg2 : (tensor<32xf32>) -> tensor<32xf32>
    %2 = tosa.rsqrt %1 : (tensor<32xf32>) -> tensor<32xf32>
    %3 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<32xf32>) -> tensor<1xf32>
    %4 = tosa.log %3 : (tensor<1xf32>) -> tensor<1xf32>
    %5 = tosa.sub %1, %1 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %6 = tosa.ceil %5 : (tensor<32xf32>) -> tensor<32xf32>
    %7 = tosa.log %4 : (tensor<1xf32>) -> tensor<1xf32>
    return %0, %6, %7 : tensor<98xi32>, tensor<32xf32>, tensor<1xf32>
  }
}
