module {
  func.func @main(%arg0: tensor<44x21x54xi1>, %arg1: tensor<44x21x54xi1>, %arg2: tensor<19x20x70x56xf32>) -> (tensor<44x21x54xi1>, tensor<20x70x56xi32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<44x21x54xi1>, tensor<44x21x54xi1>) -> tensor<44x21x54xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<44x21x54xi1>, tensor<44x21x54xi1>) -> tensor<44x21x54xi1>
    %2 = tosa.log %arg2 : (tensor<19x20x70x56xf32>) -> tensor<19x20x70x56xf32>
    %3 = tosa.greater %2, %2 : (tensor<19x20x70x56xf32>, tensor<19x20x70x56xf32>) -> tensor<19x20x70x56xi1>
    %4 = tosa.argmax %3 {axis = 0 : i32} : (tensor<19x20x70x56xi1>) -> tensor<20x70x56xi32>
    return %1, %4 : tensor<44x21x54xi1>, tensor<20x70x56xi32>
  }
}
