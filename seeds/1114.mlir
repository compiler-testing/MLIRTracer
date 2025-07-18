module {
  func.func @main(%arg0: tensor<25x10x52xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<25x10x1xf32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<25x10x52xf32>, tensor<1x1x1xf32>) -> tensor<25x10x52xf32>
    %1 = tosa.maximum %0, %0 : (tensor<25x10x52xf32>, tensor<25x10x52xf32>) -> tensor<25x10x52xf32>
    %2 = tosa.reduce_min %1 {axis = 2 : i32} : (tensor<25x10x52xf32>) -> tensor<25x10x1xf32>
    %3 = tosa.reduce_sum %2 {axis = 2 : i32} : (tensor<25x10x1xf32>) -> tensor<25x10x1xf32>
    return %3 : tensor<25x10x1xf32>
  }
}
