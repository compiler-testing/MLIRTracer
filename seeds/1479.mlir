module {
  func.func @main(%arg0: tensor<26x92x63x2xi1>, %arg1: tensor<62xf32>) -> (tensor<62xf32>, tensor<26x1x126x2xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 3, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<26x92x63x2xi1>, !tosa.shape<4>) -> tensor<26x276x126x2xi1>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<26x276x126x2xi1>) -> tensor<26x1x126x2xi1>
    %2 = tosa.bitwise_and %1, %1 : (tensor<26x1x126x2xi1>, tensor<26x1x126x2xi1>) -> tensor<26x1x126x2xi1>
    %3 = tosa.rsqrt %arg1 : (tensor<62xf32>) -> tensor<62xf32>
    %4 = tosa.bitwise_not %2 : (tensor<26x1x126x2xi1>) -> tensor<26x1x126x2xi1>
    return %3, %4 : tensor<62xf32>, tensor<26x1x126x2xi1>
  }
}
