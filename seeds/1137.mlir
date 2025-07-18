module {
  func.func @main(%arg0: tensor<56x10x24x83xi32>, %arg1: tensor<1x1x24x83xi32>, %arg2: tensor<3x33xf32>) -> (tensor<56x10x24x83xi32>, tensor<3x33xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<56x10x24x83xi32>, tensor<1x1x24x83xi32>) -> tensor<56x10x24x83xi32>
    %1 = tosa.tanh %arg2 : (tensor<3x33xf32>) -> tensor<3x33xf32>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<3x33xf32>) -> tensor<3x33xf32>
    %3 = tosa.pow %2, %2 : (tensor<3x33xf32>, tensor<3x33xf32>) -> tensor<3x33xf32>
    return %0, %3 : tensor<56x10x24x83xi32>, tensor<3x33xf32>
  }
}
