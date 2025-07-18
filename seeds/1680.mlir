module {
  func.func @main(%arg0: tensor<52x84x92x41x51x79xi16>, %arg1: tensor<57x77x29x39x93xi1>, %arg2: tensor<57x1x1x1x1xi1>, %arg3: tensor<14xi64>, %arg4: tensor<76x62x67x55x18x64xf32>) -> (tensor<52x84x92x41x51x79xi16>, tensor<1xi64>, tensor<76x62x67x55x18x64xf32>, tensor<57x77x29x39x93xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<52x84x92x41x51x79xi16>) -> tensor<52x84x92x41x51x79xi16>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<52x84x92x41x51x79xi16>, tensor<52x84x92x41x51x79xi16>) -> tensor<52x84x92x41x51x79xi16>
    %2 = tosa.identity %1 : (tensor<52x84x92x41x51x79xi16>) -> tensor<52x84x92x41x51x79xi16>
    %3 = tosa.logical_and %arg1, %arg2 : (tensor<57x77x29x39x93xi1>, tensor<57x1x1x1x1xi1>) -> tensor<57x77x29x39x93xi1>
    %4 = tosa.sub %3, %3 : (tensor<57x77x29x39x93xi1>, tensor<57x77x29x39x93xi1>) -> tensor<57x77x29x39x93xi1>
    %5 = tosa.abs %2 : (tensor<52x84x92x41x51x79xi16>) -> tensor<52x84x92x41x51x79xi16>
    %6 = tosa.reduce_max %arg3 {axis = 0 : i32} : (tensor<14xi64>) -> tensor<1xi64>
    %7 = tosa.floor %arg4 : (tensor<76x62x67x55x18x64xf32>) -> tensor<76x62x67x55x18x64xf32>
    %8 = tosa.logical_and %4, %4 : (tensor<57x77x29x39x93xi1>, tensor<57x77x29x39x93xi1>) -> tensor<57x77x29x39x93xi1>
    return %5, %6, %7, %8 : tensor<52x84x92x41x51x79xi16>, tensor<1xi64>, tensor<76x62x67x55x18x64xf32>, tensor<57x77x29x39x93xi1>
  }
}
