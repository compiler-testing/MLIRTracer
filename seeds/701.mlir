module {
  func.func @main(%arg0: tensor<29x6xi32>, %arg1: tensor<96x35x25x18xf32>, %arg2: tensor<96x1x1x18xf32>) -> (tensor<1x6xi32>, tensor<96x35x25x18xf32>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<29x6xi32>) -> tensor<1x6xi32>
    %1 = tosa.pow %arg1, %arg2 : (tensor<96x35x25x18xf32>, tensor<96x1x1x18xf32>) -> tensor<96x35x25x18xf32>
    return %0, %1 : tensor<1x6xi32>, tensor<96x35x25x18xf32>
  }
}
