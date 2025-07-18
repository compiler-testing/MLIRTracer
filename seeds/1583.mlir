module {
  func.func @main(%arg0: tensor<60x47x89x67xi8>, %arg1: tensor<58x70x11x13x72xf32>, %arg2: tensor<88x77x19x25xi1>) -> (tensor<58x70x11x13x72xf32>, tensor<1x77x19x25xi1>, tensor<1x47x89x67xi8>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<60x47x89x67xi8>) -> tensor<1x47x89x67xi8>
    %1 = tosa.clz %0 : (tensor<1x47x89x67xi8>) -> tensor<1x47x89x67xi8>
    %2 = tosa.rsqrt %arg1 : (tensor<58x70x11x13x72xf32>) -> tensor<58x70x11x13x72xf32>
    %3 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<88x77x19x25xi1>) -> tensor<1x77x19x25xi1>
    %4 = tosa.bitwise_xor %1, %1 : (tensor<1x47x89x67xi8>, tensor<1x47x89x67xi8>) -> tensor<1x47x89x67xi8>
    return %2, %3, %4 : tensor<58x70x11x13x72xf32>, tensor<1x77x19x25xi1>, tensor<1x47x89x67xi8>
  }
}
