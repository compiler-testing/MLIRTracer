module {
  func.func @main(%arg0: tensor<93x12x63x19xi32>, %arg1: tensor<1x1x1x1xi32>, %arg2: tensor<91xf32>) -> (tensor<93x12x63x19xi32>, tensor<91xf32>, tensor<1xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<93x12x63x19xi32>, tensor<1x1x1x1xi32>) -> tensor<93x12x63x19xi32>
    %1 = tosa.sigmoid %arg2 : (tensor<91xf32>) -> tensor<91xf32>
    %2 = tosa.sigmoid %1 : (tensor<91xf32>) -> tensor<91xf32>
    %3 = tosa.rsqrt %1 : (tensor<91xf32>) -> tensor<91xf32>
    %4 = tosa.exp %2 : (tensor<91xf32>) -> tensor<91xf32>
    %5 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<91xf32>) -> tensor<1xf32>
    return %0, %3, %5 : tensor<93x12x63x19xi32>, tensor<91xf32>, tensor<1xf32>
  }
}
