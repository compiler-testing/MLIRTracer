module {
  func.func @main(%arg0: tensor<91x3x95x100x16x53xf32>, %arg1: tensor<91x1x95x1x16x1xf32>, %arg2: tensor<4x65xi16>) -> (tensor<65xi32>, tensor<91x3x95x100x16x53xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<91x3x95x100x16x53xf32>, tensor<91x1x95x1x16x1xf32>) -> tensor<91x3x95x100x16x53xf32>
    %1 = tosa.argmax %arg2 {axis = 0 : i32} : (tensor<4x65xi16>) -> tensor<65xi32>
    %2 = tosa.exp %0 : (tensor<91x3x95x100x16x53xf32>) -> tensor<91x3x95x100x16x53xf32>
    return %1, %2 : tensor<65xi32>, tensor<91x3x95x100x16x53xf32>
  }
}
