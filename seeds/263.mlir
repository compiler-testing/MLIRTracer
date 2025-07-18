module {
  func.func @main(%arg0: tensor<30x25xi64>, %arg1: tensor<56x18xf32>, %arg2: tensor<47x55xi1>, %arg3: tensor<47x55xi1>) -> (tensor<30x25xi64>, tensor<56x18xf32>, tensor<47x55xi1>, tensor<47x1xi1>, tensor<47x55xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<30x25xi64>, !tosa.shape<2>) -> tensor<30x25xi64>
    %1 = tosa.floor %arg1 : (tensor<56x18xf32>) -> tensor<56x18xf32>
    %2 = tosa.add %1, %1 : (tensor<56x18xf32>, tensor<56x18xf32>) -> tensor<56x18xf32>
    %3 = tosa.logical_or %arg2, %arg3 : (tensor<47x55xi1>, tensor<47x55xi1>) -> tensor<47x55xi1>
    %4 = tosa.bitwise_and %0, %0 : (tensor<30x25xi64>, tensor<30x25xi64>) -> tensor<30x25xi64>
    %5 = tosa.add %4, %0 : (tensor<30x25xi64>, tensor<30x25xi64>) -> tensor<30x25xi64>
    %6 = tosa.exp %2 : (tensor<56x18xf32>) -> tensor<56x18xf32>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %7 = tosa.negate %3, %in_zp_7, %out_zp_7 : (tensor<47x55xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<47x55xi1>
    %8 = tosa.reduce_product %3 {axis = 1 : i32} : (tensor<47x55xi1>) -> tensor<47x1xi1>
    %9 = tosa.logical_not %3 : (tensor<47x55xi1>) -> tensor<47x55xi1>
    return %5, %6, %7, %8, %9 : tensor<30x25xi64>, tensor<56x18xf32>, tensor<47x55xi1>, tensor<47x1xi1>, tensor<47x55xi1>
  }
}
