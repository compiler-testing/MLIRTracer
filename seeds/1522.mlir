module {
  func.func @main(%arg0: tensor<25x68x99x65x88xi1>, %arg1: tensor<1x68x99x65x88xi1>, %arg2: tensor<24x2xi1>) -> (tensor<25x68x99x65x88xi1>, tensor<72x1xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<25x68x99x65x88xi1>, tensor<1x68x99x65x88xi1>) -> tensor<25x68x99x65x88xi1>
    %1 = tosa.logical_or %0, %0 : (tensor<25x68x99x65x88xi1>, tensor<25x68x99x65x88xi1>) -> tensor<25x68x99x65x88xi1>
    %2 = tosa.clz %1 : (tensor<25x68x99x65x88xi1>) -> tensor<25x68x99x65x88xi1>
    %3 = tosa.logical_and %2, %0 : (tensor<25x68x99x65x88xi1>, tensor<25x68x99x65x88xi1>) -> tensor<25x68x99x65x88xi1>
    %4 = tosa.logical_right_shift %3, %2 : (tensor<25x68x99x65x88xi1>, tensor<25x68x99x65x88xi1>) -> tensor<25x68x99x65x88xi1>
    %5 = tosa.reduce_all %arg2 {axis = 1 : i32} : (tensor<24x2xi1>) -> tensor<24x1xi1>
    %6 = tosa.sub %4, %3 : (tensor<25x68x99x65x88xi1>, tensor<25x68x99x65x88xi1>) -> tensor<25x68x99x65x88xi1>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %7 = tosa.negate %6, %in_zp_7, %out_zp_7 : (tensor<25x68x99x65x88xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<25x68x99x65x88xi1>
    %8 = tosa.bitwise_or %5, %5 : (tensor<24x1xi1>, tensor<24x1xi1>) -> tensor<24x1xi1>
    %t_9 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.tile %8, %t_9 : (tensor<24x1xi1>, !tosa.shape<2>) -> tensor<72x1xi1>
    return %7, %9 : tensor<25x68x99x65x88xi1>, tensor<72x1xi1>
  }
}
