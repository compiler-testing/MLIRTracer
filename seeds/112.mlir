module {
  func.func @main(%arg0: tensor<68xi32>, %arg1: tensor<68xi32>, %arg2: tensor<58x1x8x81xf32>, %arg3: tensor<58x1x1x81xf32>) -> (tensor<68xi32>, tensor<58x1x1x81xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<68xi32>, tensor<68xi32>) -> tensor<68xi32>
    %1 = tosa.pow %arg2, %arg3 : (tensor<58x1x8x81xf32>, tensor<58x1x1x81xf32>) -> tensor<58x1x8x81xf32>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<58x1x8x81xf32>) -> tensor<58x1x1x81xf32>
    return %0, %2 : tensor<68xi32>, tensor<58x1x1x81xf32>
  }
}
