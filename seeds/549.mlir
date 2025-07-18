module {
  func.func @main(%arg0: tensor<26xi32>, %arg1: tensor<39xi32>) -> tensor<65xi32> {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<26xi32>, tensor<39xi32>) -> tensor<65xi32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<65xi32>, tensor<65xi32>) -> tensor<65xi32>
    return %1 : tensor<65xi32>
  }
}
