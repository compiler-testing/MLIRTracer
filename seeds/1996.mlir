module {
  func.func @main(%arg0: tensor<53x75x4xi8>, %arg1: tensor<53x1x1xi8>, %arg2: tensor<32x35xi8>, %arg3: tensor<1x1xi8>, %arg4: tensor<20x67x14xi64>, %arg5: tensor<1x1x14xi64>, %arg6: tensor<79x70xf32>, %arg7: tensor<1x1xf32>, %arg8: tensor<25xi32>, %arg9: tensor<1xi32>, %arg10: tensor<16x100x59x59xf32>) -> (tensor<53x75x4xi1>, tensor<32x35xi1>, tensor<20x67x14xi1>, tensor<79x70xi1>, tensor<25xi1>, tensor<16x100x59x59xi1>, tensor<25xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<53x75x4xi8>, tensor<53x1x1xi8>) -> tensor<53x75x4xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<53x75x4xi1>, tensor<53x75x4xi1>) -> tensor<53x75x4xi1>
    %2 = tosa.equal %arg2, %arg3 : (tensor<32x35xi8>, tensor<1x1xi8>) -> tensor<32x35xi1>
    %3 = tosa.equal %arg4, %arg5 : (tensor<20x67x14xi64>, tensor<1x1x14xi64>) -> tensor<20x67x14xi1>
    %4 = tosa.equal %arg6, %arg7 : (tensor<79x70xf32>, tensor<1x1xf32>) -> tensor<79x70xi1>
    %5 = tosa.greater_equal %arg8, %arg9 : (tensor<25xi32>, tensor<1xi32>) -> tensor<25xi1>
    %6 = tosa.logical_xor %5, %5 : (tensor<25xi1>, tensor<25xi1>) -> tensor<25xi1>
    %7 = tosa.bitwise_and %6, %6 : (tensor<25xi1>, tensor<25xi1>) -> tensor<25xi1>
    %8 = tosa.tanh %arg10 : (tensor<16x100x59x59xf32>) -> tensor<16x100x59x59xf32>
    %9 = tosa.equal %8, %8 : (tensor<16x100x59x59xf32>, tensor<16x100x59x59xf32>) -> tensor<16x100x59x59xi1>
    %10 = tosa.bitwise_and %5, %5 : (tensor<25xi1>, tensor<25xi1>) -> tensor<25xi1>
    return %1, %2, %3, %4, %7, %9, %10 : tensor<53x75x4xi1>, tensor<32x35xi1>, tensor<20x67x14xi1>, tensor<79x70xi1>, tensor<25xi1>, tensor<16x100x59x59xi1>, tensor<25xi1>
  }
}
