module {
  func.func @main(%arg0: tensor<15xi64>, %arg1: tensor<96x51xf32>, %arg2: tensor<1x51xf32>, %arg3: tensor<35x7x25x50xi1>, %arg4: tensor<35x1x25x1xi1>) -> (tensor<1xi64>, tensor<35x7x25x50xi1>, tensor<96x51xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<15xi64>) -> tensor<1xi64>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<1xi64>
    %3 = tosa.pow %arg1, %arg2 : (tensor<96x51xf32>, tensor<1x51xf32>) -> tensor<96x51xf32>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4 = tosa.negate %2, %in_zp_4, %out_zp_4 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %5 = tosa.logical_right_shift %4, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %6 = tosa.logical_or %arg3, %arg4 : (tensor<35x7x25x50xi1>, tensor<35x1x25x1xi1>) -> tensor<35x7x25x50xi1>
    %7 = tosa.bitwise_xor %5, %1 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %8 = tosa.add %7, %4 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %9 = tosa.bitwise_xor %8, %5 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %10 = tosa.tanh %3 : (tensor<96x51xf32>) -> tensor<96x51xf32>
    %11 = tosa.floor %10 : (tensor<96x51xf32>) -> tensor<96x51xf32>
    %12 = tosa.abs %6 : (tensor<35x7x25x50xi1>) -> tensor<35x7x25x50xi1>
    %13 = tosa.pow %11, %3 : (tensor<96x51xf32>, tensor<96x51xf32>) -> tensor<96x51xf32>
    return %9, %12, %13 : tensor<1xi64>, tensor<35x7x25x50xi1>, tensor<96x51xf32>
  }
}
