module {
  func.func @main(%arg0: tensor<7x70x92x49x68xi32>, %arg1: tensor<7x70x20x49x68xi32>, %arg2: tensor<95xf32>) -> (tensor<7x70x112x49x68xi32>, tensor<95xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<7x70x92x49x68xi32>, tensor<7x70x20x49x68xi32>) -> tensor<7x70x112x49x68xi32>
    %1 = tosa.rsqrt %arg2 : (tensor<95xf32>) -> tensor<95xf32>
    %2 = tosa.sub %0, %0 : (tensor<7x70x112x49x68xi32>, tensor<7x70x112x49x68xi32>) -> tensor<7x70x112x49x68xi32>
    %3 = tosa.log %1 : (tensor<95xf32>) -> tensor<95xf32>
    return %2, %3 : tensor<7x70x112x49x68xi32>, tensor<95xf32>
  }
}
