module {
  func.func @main(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<55x41xi16>, %arg3: tensor<1x1xi16>) -> (tensor<32xf32>, tensor<41xi32>, tensor<32xf32>, tensor<32xi1>, tensor<1xi32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %1 = tosa.log %0 : (tensor<32xf32>) -> tensor<32xf32>
    %2 = tosa.logical_right_shift %arg2, %arg3 : (tensor<55x41xi16>, tensor<1x1xi16>) -> tensor<55x41xi16>
    %3 = tosa.exp %1 : (tensor<32xf32>) -> tensor<32xf32>
    %4 = tosa.add %3, %1 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %5 = tosa.argmax %2 {axis = 0 : i32} : (tensor<55x41xi16>) -> tensor<41xi32>
    %6 = tosa.greater_equal %4, %1 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xi1>
    %7 = tosa.exp %3 : (tensor<32xf32>) -> tensor<32xf32>
    %8 = tosa.minimum %5, %5 : (tensor<41xi32>, tensor<41xi32>) -> tensor<41xi32>
    %9 = tosa.reduce_min %8 {axis = 0 : i32} : (tensor<41xi32>) -> tensor<1xi32>
    %10 = tosa.arithmetic_right_shift %5, %5 {round = true} : (tensor<41xi32>, tensor<41xi32>) -> tensor<41xi32>
    %11 = tosa.rsqrt %1 : (tensor<32xf32>) -> tensor<32xf32>
    %12 = tosa.logical_xor %6, %6 : (tensor<32xi1>, tensor<32xi1>) -> tensor<32xi1>
    %13 = tosa.logical_not %12 : (tensor<32xi1>) -> tensor<32xi1>
    %14 = tosa.arithmetic_right_shift %9, %9 {round = true} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %7, %10, %11, %13, %14 : tensor<32xf32>, tensor<41xi32>, tensor<32xf32>, tensor<32xi1>, tensor<1xi32>
  }
}
