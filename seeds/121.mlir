module {
  func.func @main(%arg0: tensor<41x48x50x62x33x85xf32>, %arg1: tensor<56x26x25x23xi32>, %arg2: tensor<56x26x25x23xi32>) -> (tensor<41x48x50x62x33x85xf32>, tensor<56x26x25x23xi32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<41x48x50x62x33x85xf32>) -> tensor<41x48x50x62x33x85xf32>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<56x26x25x23xi32>, tensor<56x26x25x23xi32>) -> tensor<56x26x25x23xi32>
    return %0, %1 : tensor<41x48x50x62x33x85xf32>, tensor<56x26x25x23xi32>
  }
}
