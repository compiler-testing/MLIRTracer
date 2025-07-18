module {
  func.func @main(%arg0: tensor<12x83x85xf32>, %arg1: tensor<13x69x59x72x54xi32>, %arg2: tensor<1x1x1x72x54xi32>) -> (tensor<13x69x59x72x54xi32>, tensor<1x83x1xf32>) {
    %0 = tosa.reduce_product %arg0 {axis = 2 : i32} : (tensor<12x83x85xf32>) -> tensor<12x83x1xf32>
    %1 = tosa.minimum %0, %0 : (tensor<12x83x1xf32>, tensor<12x83x1xf32>) -> tensor<12x83x1xf32>
    %2 = tosa.intdiv %arg1, %arg2 : (tensor<13x69x59x72x54xi32>, tensor<1x1x1x72x54xi32>) -> tensor<13x69x59x72x54xi32>
    %3 = tosa.pow %1, %0 : (tensor<12x83x1xf32>, tensor<12x83x1xf32>) -> tensor<12x83x1xf32>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<12x83x1xf32>) -> tensor<1x83x1xf32>
    return %2, %4 : tensor<13x69x59x72x54xi32>, tensor<1x83x1xf32>
  }
}
