module {
  func.func @main(%arg0: tensor<92x2x54x46x56xf32>, %arg1: tensor<1x1x54x46x1xf32>, %arg2: tensor<46x5xi32>, %arg3: tensor<1x1xi32>) -> (tensor<92x2x54x46x56xf32>, tensor<46x1xi32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<92x2x54x46x56xf32>, tensor<1x1x54x46x1xf32>) -> tensor<92x2x54x46x56xf32>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<46x5xi32>, tensor<1x1xi32>) -> tensor<46x5xi32>
    %2 = tosa.tanh %0 : (tensor<92x2x54x46x56xf32>) -> tensor<92x2x54x46x56xf32>
    %3 = tosa.reduce_min %1 {axis = 1 : i32} : (tensor<46x5xi32>) -> tensor<46x1xi32>
    return %2, %3 : tensor<92x2x54x46x56xf32>, tensor<46x1xi32>
  }
}
