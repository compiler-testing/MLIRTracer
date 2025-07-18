module {
  func.func @main(%arg0: tensor<37x61x50x98xi1>, %arg1: tensor<89x33xf32>) -> (tensor<37x1x1x98xi1>, tensor<89x33xf32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<37x61x50x98xi1>) -> tensor<37x61x1x98xi1>
    %1 = tosa.reduce_max %0 {axis = 2 : i32} : (tensor<37x61x1x98xi1>) -> tensor<37x61x1x98xi1>
    %2 = tosa.reciprocal %arg1 : (tensor<89x33xf32>) -> tensor<89x33xf32>
    %3 = tosa.sub %2, %2 : (tensor<89x33xf32>, tensor<89x33xf32>) -> tensor<89x33xf32>
    %4 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<37x61x1x98xi1>) -> tensor<37x1x1x98xi1>
    %5 = tosa.identity %3 : (tensor<89x33xf32>) -> tensor<89x33xf32>
    %6 = tosa.tanh %5 : (tensor<89x33xf32>) -> tensor<89x33xf32>
    return %4, %6 : tensor<37x1x1x98xi1>, tensor<89x33xf32>
  }
}
