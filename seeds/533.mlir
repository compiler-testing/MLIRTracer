module {
  func.func @main(%arg0: tensor<90x48x32x41x39xi1>, %arg1: tensor<90x1x1x41x39xi1>, %arg2: tensor<51x35x34x97xi1>, %arg3: tensor<50x69xf32>) -> (tensor<90x48x32x41x39xi1>, tensor<1x1x34x97xi1>, tensor<51x1x1x97xi1>, tensor<50x69xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<90x48x32x41x39xi1>, tensor<90x1x1x41x39xi1>) -> tensor<90x48x32x41x39xi1>
    %1 = tosa.logical_or %0, %0 : (tensor<90x48x32x41x39xi1>, tensor<90x48x32x41x39xi1>) -> tensor<90x48x32x41x39xi1>
    %2 = tosa.reduce_any %arg2 {axis = 1 : i32} : (tensor<51x35x34x97xi1>) -> tensor<51x1x34x97xi1>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<51x1x34x97xi1>) -> tensor<51x1x34x97xi1>
    %4 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<51x1x34x97xi1>) -> tensor<1x1x34x97xi1>
    %5 = tosa.reduce_all %3 {axis = 2 : i32} : (tensor<51x1x34x97xi1>) -> tensor<51x1x1x97xi1>
    %6 = tosa.tanh %arg3 : (tensor<50x69xf32>) -> tensor<50x69xf32>
    return %1, %4, %5, %6 : tensor<90x48x32x41x39xi1>, tensor<1x1x34x97xi1>, tensor<51x1x1x97xi1>, tensor<50x69xf32>
  }
}
