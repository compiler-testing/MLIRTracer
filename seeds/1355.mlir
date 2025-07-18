module {
  func.func @main(%arg0: tensor<100xi1>, %arg1: tensor<40x72x66x35xf32>) -> (tensor<1xi1>, tensor<1xi1>, tensor<80x72x132x70xf32>, tensor<1xi1>, tensor<2x144x66x35xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<100xi1>) -> tensor<1xi1>
    %1 = tosa.floor %arg1 : (tensor<40x72x66x35xf32>) -> tensor<40x72x66x35xf32>
    %2 = tosa.sigmoid %1 : (tensor<40x72x66x35xf32>) -> tensor<40x72x66x35xf32>
    %3 = tosa.concat %1, %2 {axis = 1 : i32} : (tensor<40x72x66x35xf32>, tensor<40x72x66x35xf32>) -> tensor<40x144x66x35xf32>
    %4 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<40x144x66x35xf32>) -> tensor<1x144x66x35xf32>
    %5 = tosa.pow %2, %1 : (tensor<40x72x66x35xf32>, tensor<40x72x66x35xf32>) -> tensor<40x72x66x35xf32>
    %6 = tosa.clz %0 : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.log %4 : (tensor<1x144x66x35xf32>) -> tensor<1x144x66x35xf32>
    %8 = tosa.sigmoid %5 : (tensor<40x72x66x35xf32>) -> tensor<40x72x66x35xf32>
    %9 = tosa.sub %8, %8 : (tensor<40x72x66x35xf32>, tensor<40x72x66x35xf32>) -> tensor<40x72x66x35xf32>
    %10 = tosa.ceil %9 : (tensor<40x72x66x35xf32>) -> tensor<40x72x66x35xf32>
    %11 = tosa.bitwise_or %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.abs %10 : (tensor<40x72x66x35xf32>) -> tensor<40x72x66x35xf32>
    %13 = tosa.logical_or %0, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.tanh %12 : (tensor<40x72x66x35xf32>) -> tensor<40x72x66x35xf32>
    %15 = tosa.concat %14, %8 {axis = 2 : i32} : (tensor<40x72x66x35xf32>, tensor<40x72x66x35xf32>) -> tensor<40x72x132x35xf32>
    %16 = tosa.concat %7, %7 {axis = 0 : i32} : (tensor<1x144x66x35xf32>, tensor<1x144x66x35xf32>) -> tensor<2x144x66x35xf32>
    %17 = tosa.clz %0 : (tensor<1xi1>) -> tensor<1xi1>
    %18 = tosa.bitwise_and %11, %17 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %t_19 = tosa.const_shape {values = dense<[ 2, 1, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %19 = tosa.tile %15, %t_19 : (tensor<40x72x132x35xf32>, !tosa.shape<4>) -> tensor<80x72x132x70xf32>
    %20 = tosa.add %16, %16 : (tensor<2x144x66x35xf32>, tensor<2x144x66x35xf32>) -> tensor<2x144x66x35xf32>
    %21 = tosa.maximum %19, %19 : (tensor<80x72x132x70xf32>, tensor<80x72x132x70xf32>) -> tensor<80x72x132x70xf32>
    %22 = tosa.reduce_any %11 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %23 = tosa.greater %20, %16 : (tensor<2x144x66x35xf32>, tensor<2x144x66x35xf32>) -> tensor<2x144x66x35xi1>
    %24 = tosa.bitwise_not %23 : (tensor<2x144x66x35xi1>) -> tensor<2x144x66x35xi1>
    return %13, %18, %21, %22, %24 : tensor<1xi1>, tensor<1xi1>, tensor<80x72x132x70xf32>, tensor<1xi1>, tensor<2x144x66x35xi1>
  }
}
