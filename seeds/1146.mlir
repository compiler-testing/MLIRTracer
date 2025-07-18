module {
  func.func @main(%arg0: tensor<26x81x88xi1>, %arg1: tensor<25x75x19x92x79xf32>, %arg2: tensor<1x75x19x92x79xf32>) -> (tensor<26x1x1xi1>, tensor<25x75x19x92x79xf32>, tensor<26x81x1xi1>, tensor<25x75x19x184x79xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<26x81x88xi1>) -> tensor<26x81x1xi1>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<26x81x1xi1>) -> tensor<26x1x1xi1>
    %2 = tosa.maximum %arg1, %arg2 : (tensor<25x75x19x92x79xf32>, tensor<1x75x19x92x79xf32>) -> tensor<25x75x19x92x79xf32>
    %3 = tosa.maximum %2, %2 : (tensor<25x75x19x92x79xf32>, tensor<25x75x19x92x79xf32>) -> tensor<25x75x19x92x79xf32>
    %4 = tosa.maximum %2, %3 : (tensor<25x75x19x92x79xf32>, tensor<25x75x19x92x79xf32>) -> tensor<25x75x19x92x79xf32>
    %5 = tosa.bitwise_not %0 : (tensor<26x81x1xi1>) -> tensor<26x81x1xi1>
    %6 = tosa.reciprocal %3 : (tensor<25x75x19x92x79xf32>) -> tensor<25x75x19x92x79xf32>
    %7 = tosa.sub %6, %2 : (tensor<25x75x19x92x79xf32>, tensor<25x75x19x92x79xf32>) -> tensor<25x75x19x92x79xf32>
    %8 = tosa.bitwise_not %5 : (tensor<26x81x1xi1>) -> tensor<26x81x1xi1>
    %9 = tosa.concat %7, %6 {axis = 3 : i32} : (tensor<25x75x19x92x79xf32>, tensor<25x75x19x92x79xf32>) -> tensor<25x75x19x184x79xf32>
    %10 = tosa.log %9 : (tensor<25x75x19x184x79xf32>) -> tensor<25x75x19x184x79xf32>
    return %1, %4, %8, %10 : tensor<26x1x1xi1>, tensor<25x75x19x92x79xf32>, tensor<26x81x1xi1>, tensor<25x75x19x184x79xf32>
  }
}
