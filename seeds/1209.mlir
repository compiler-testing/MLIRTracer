module {
  func.func @main(%arg0: tensor<44x24x3x68x75xi16>, %arg1: tensor<44x24x1x68x75xi16>, %arg2: tensor<13x75x62xi1>, %arg3: tensor<50xf32>) -> (tensor<44x24x3x68x75xi16>, tensor<1x1xi32>, tensor<3x225x186xi1>, tensor<50xi1>, tensor<50xf32>, tensor<1x75x62xi1>, tensor<1x75x1xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<44x24x3x68x75xi16>, tensor<44x24x1x68x75xi16>) -> tensor<44x24x3x68x75xi16>
    %1 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<13x75x62xi1>) -> tensor<1x75x62xi1>
    %2 = tosa.reduce_min %1 {axis = 2 : i32} : (tensor<1x75x62xi1>) -> tensor<1x75x1xi1>
    %3 = tosa.reverse %1 {axis = 1 : i32} : (tensor<1x75x62xi1>) -> tensor<1x75x62xi1>
    %4 = tosa.argmax %2 {axis = 1 : i32} : (tensor<1x75x1xi1>) -> tensor<1x1xi32>
    %t_5 = tosa.const_shape {values = dense<[ 3, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.tile %3, %t_5 : (tensor<1x75x62xi1>, !tosa.shape<3>) -> tensor<3x225x186xi1>
    %6 = tosa.bitwise_not %4 : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %7 = tosa.logical_or %5, %5 : (tensor<3x225x186xi1>, tensor<3x225x186xi1>) -> tensor<3x225x186xi1>
    %8 = tosa.logical_left_shift %5, %7 : (tensor<3x225x186xi1>, tensor<3x225x186xi1>) -> tensor<3x225x186xi1>
    %9 = tosa.floor %arg3 : (tensor<50xf32>) -> tensor<50xf32>
    %10 = tosa.log %9 : (tensor<50xf32>) -> tensor<50xf32>
    %11 = tosa.greater %9, %9 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xi1>
    %12 = tosa.pow %9, %10 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xf32>
    %13 = tosa.bitwise_xor %3, %1 : (tensor<1x75x62xi1>, tensor<1x75x62xi1>) -> tensor<1x75x62xi1>
    %14 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<1x75x1xi1>) -> tensor<1x75x1xi1>
    return %0, %6, %8, %11, %12, %13, %14 : tensor<44x24x3x68x75xi16>, tensor<1x1xi32>, tensor<3x225x186xi1>, tensor<50xi1>, tensor<50xf32>, tensor<1x75x62xi1>, tensor<1x75x1xi1>
  }
}
