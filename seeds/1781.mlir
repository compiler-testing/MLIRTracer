module {
  func.func @main(%arg0: tensor<66xi64>, %arg1: tensor<25x68xf32>, %arg2: tensor<53xi1>) -> (tensor<6xi64>, tensor<25x68xf32>, tensor<1xi1>, tensor<2xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 6 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<66xi64>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<6xi64>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<6xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<6xi64>
    %2 = tosa.add %1, %1 : (tensor<6xi64>, tensor<6xi64>) -> tensor<6xi64>
    %3 = tosa.arithmetic_right_shift %2, %1 {round = false} : (tensor<6xi64>, tensor<6xi64>) -> tensor<6xi64>
    %4 = tosa.ceil %arg1 : (tensor<25x68xf32>) -> tensor<25x68xf32>
    %5 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<53xi1>) -> tensor<1xi1>
    %6 = tosa.sub %5, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.concat %5, %5 {axis = 0 : i32} : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
    return %3, %4, %6, %7 : tensor<6xi64>, tensor<25x68xf32>, tensor<1xi1>, tensor<2xi1>
  }
}
