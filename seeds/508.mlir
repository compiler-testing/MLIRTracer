module {
  func.func @main(%arg0: tensor<62x40x13x90xi32>, %arg1: tensor<1x1x1x1xi32>, %arg2: tensor<4xf32>, %arg3: tensor<46x57x53x97x49xi1>, %arg4: tensor<1x57x1x1x49xi1>) -> (tensor<62x40x13x90xi32>, tensor<46x57x53x97x49xi1>, tensor<1xf32>, tensor<1xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<62x40x13x90xi32>, tensor<1x1x1x1xi32>) -> tensor<62x40x13x90xi32>
    %1 = tosa.tanh %arg2 : (tensor<4xf32>) -> tensor<4xf32>
    %2 = tosa.logical_and %arg3, %arg4 : (tensor<46x57x53x97x49xi1>, tensor<1x57x1x1x49xi1>) -> tensor<46x57x53x97x49xi1>
    %3 = tosa.reciprocal %1 : (tensor<4xf32>) -> tensor<4xf32>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<4xf32>) -> tensor<1xf32>
    %5 = tosa.add %2, %2 : (tensor<46x57x53x97x49xi1>, tensor<46x57x53x97x49xi1>) -> tensor<46x57x53x97x49xi1>
    %6 = tosa.greater %4, %4 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.tanh %4 : (tensor<1xf32>) -> tensor<1xf32>
    %9 = tosa.reduce_any %7 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %in_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %10 = tosa.negate %9, %in_zp_10, %out_zp_10 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %0, %5, %8, %10 : tensor<62x40x13x90xi32>, tensor<46x57x53x97x49xi1>, tensor<1xf32>, tensor<1xi1>
  }
}
