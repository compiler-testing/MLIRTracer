module {
  func.func @main(%arg0: tensor<40x54xi32>, %arg1: tensor<15xi1>, %arg2: tensor<1xi1>) -> (tensor<40x54xi32>, tensor<15xi1>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<40x54xi32>) -> tensor<40x54xi32>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<40x54xi32>, tensor<40x54xi32>) -> tensor<40x54xi32>
    %2 = tosa.logical_xor %arg1, %arg2 : (tensor<15xi1>, tensor<1xi1>) -> tensor<15xi1>
    return %1, %2 : tensor<40x54xi32>, tensor<15xi1>
  }
}
