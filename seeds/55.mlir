module {
  func.func @main(%arg0: tensor<73x44x86x42x8x77xf32>, %arg1: tensor<5xi1>, %arg2: tensor<1xi1>) -> (tensor<73x44x86x42x8x77xf32>, tensor<5xi1>, tensor<5xi1>) {
    %0 = tosa.log %arg0 : (tensor<73x44x86x42x8x77xf32>) -> tensor<73x44x86x42x8x77xf32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<5xi1>, tensor<1xi1>) -> tensor<5xi1>
    %2 = tosa.maximum %0, %0 : (tensor<73x44x86x42x8x77xf32>, tensor<73x44x86x42x8x77xf32>) -> tensor<73x44x86x42x8x77xf32>
    %3 = tosa.abs %2 : (tensor<73x44x86x42x8x77xf32>) -> tensor<73x44x86x42x8x77xf32>
    %4 = tosa.bitwise_not %1 : (tensor<5xi1>) -> tensor<5xi1>
    %5 = tosa.identity %1 : (tensor<5xi1>) -> tensor<5xi1>
    return %3, %4, %5 : tensor<73x44x86x42x8x77xf32>, tensor<5xi1>, tensor<5xi1>
  }
}
