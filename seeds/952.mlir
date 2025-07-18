module {
  func.func @main(%arg0: tensor<60x46x27x39x6xi8>, %arg1: tensor<45x28x5x67xi1>, %arg2: tensor<49xf32>) -> (tensor<60x46x27x39x6xi8>, tensor<45x1x5x1xi1>, tensor<45x1x5x67xi1>, tensor<1x1x5x67xi1>, tensor<49xf32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<60x46x27x39x6xi8>) -> tensor<60x46x27x39x6xi8>
    %1 = tosa.sub %0, %0 : (tensor<60x46x27x39x6xi8>, tensor<60x46x27x39x6xi8>) -> tensor<60x46x27x39x6xi8>
    %2 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<45x28x5x67xi1>) -> tensor<45x1x5x67xi1>
    %3 = tosa.exp %arg2 : (tensor<49xf32>) -> tensor<49xf32>
    %4 = tosa.reduce_all %2 {axis = 3 : i32} : (tensor<45x1x5x67xi1>) -> tensor<45x1x5x1xi1>
    %5 = tosa.reduce_product %2 {axis = 1 : i32} : (tensor<45x1x5x67xi1>) -> tensor<45x1x5x67xi1>
    %6 = tosa.clz %5 : (tensor<45x1x5x67xi1>) -> tensor<45x1x5x67xi1>
    %7 = tosa.reduce_sum %5 {axis = 0 : i32} : (tensor<45x1x5x67xi1>) -> tensor<1x1x5x67xi1>
    %8 = tosa.exp %3 : (tensor<49xf32>) -> tensor<49xf32>
    return %1, %4, %6, %7, %8 : tensor<60x46x27x39x6xi8>, tensor<45x1x5x1xi1>, tensor<45x1x5x67xi1>, tensor<1x1x5x67xi1>, tensor<49xf32>
  }
}
