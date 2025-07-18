module {
  func.func @main(%arg0: tensor<91x62x26x91xf32>, %arg1: tensor<62x65x36xi1>) -> (tensor<91x1x26x91xf32>, tensor<62x1x36xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<91x62x26x91xf32>) -> tensor<91x62x26x91xf32>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<91x62x26x91xf32>) -> tensor<91x1x26x91xf32>
    %2 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<62x65x36xi1>) -> tensor<62x1x36xi1>
    %3 = tosa.minimum %1, %1 : (tensor<91x1x26x91xf32>, tensor<91x1x26x91xf32>) -> tensor<91x1x26x91xf32>
    %4 = tosa.rsqrt %3 : (tensor<91x1x26x91xf32>) -> tensor<91x1x26x91xf32>
    %5 = tosa.bitwise_xor %2, %2 : (tensor<62x1x36xi1>, tensor<62x1x36xi1>) -> tensor<62x1x36xi1>
    return %4, %5 : tensor<91x1x26x91xf32>, tensor<62x1x36xi1>
  }
}
