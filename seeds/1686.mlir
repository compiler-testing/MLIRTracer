module {
  func.func @main(%arg0: tensor<14x29x26x77xf32>, %arg1: tensor<39x5xi1>) -> (tensor<14x29x26x77xf32>, tensor<1x5xi1>, tensor<1x5xi1>) {
    %0 = tosa.log %arg0 : (tensor<14x29x26x77xf32>) -> tensor<14x29x26x77xf32>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<39x5xi1>) -> tensor<1x5xi1>
    %2 = tosa.sub %1, %1 : (tensor<1x5xi1>, tensor<1x5xi1>) -> tensor<1x5xi1>
    %3 = tosa.reciprocal %0 : (tensor<14x29x26x77xf32>) -> tensor<14x29x26x77xf32>
    %4 = tosa.bitwise_xor %2, %1 : (tensor<1x5xi1>, tensor<1x5xi1>) -> tensor<1x5xi1>
    %5 = tosa.identity %2 : (tensor<1x5xi1>) -> tensor<1x5xi1>
    return %3, %4, %5 : tensor<14x29x26x77xf32>, tensor<1x5xi1>, tensor<1x5xi1>
  }
}
