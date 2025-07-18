module {
  func.func @main(%arg0: tensor<92x83x49x96xi1>, %arg1: tensor<92x1x1x96xi1>, %arg2: tensor<91x62xf32>, %arg3: tensor<44x19x93x12x57xi32>, %arg4: tensor<1x19x93x12x1xi32>) -> (tensor<1x83x1x96xi1>, tensor<44x19x93x12x57xi32>, tensor<1x83x49x96xi1>, tensor<11284xf32>, tensor<91x62xf32>, tensor<62xi32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<92x83x49x96xi1>, tensor<92x1x1x96xi1>) -> tensor<92x83x49x96xi1>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<92x83x49x96xi1>) -> tensor<1x83x49x96xi1>
    %2 = tosa.rsqrt %arg2 : (tensor<91x62xf32>) -> tensor<91x62xf32>
    %3 = tosa.reduce_all %1 {axis = 2 : i32} : (tensor<1x83x49x96xi1>) -> tensor<1x83x1x96xi1>
    %4 = tosa.intdiv %arg3, %arg4 : (tensor<44x19x93x12x57xi32>, tensor<1x19x93x12x1xi32>) -> tensor<44x19x93x12x57xi32>
    %5 = tosa.floor %2 : (tensor<91x62xf32>) -> tensor<91x62xf32>
    %6 = tosa.sub %2, %5 : (tensor<91x62xf32>, tensor<91x62xf32>) -> tensor<91x62xf32>
    %7 = tosa.logical_right_shift %4, %4 : (tensor<44x19x93x12x57xi32>, tensor<44x19x93x12x57xi32>) -> tensor<44x19x93x12x57xi32>
    %8 = tosa.tanh %2 : (tensor<91x62xf32>) -> tensor<91x62xf32>
    %9 = tosa.arithmetic_right_shift %1, %1 {round = false} : (tensor<1x83x49x96xi1>, tensor<1x83x49x96xi1>) -> tensor<1x83x49x96xi1>
    %10 = tosa.ceil %8 : (tensor<91x62xf32>) -> tensor<91x62xf32>
    %11 = tosa.argmax %5 {axis = 0 : i32} : (tensor<91x62xf32>) -> tensor<62xi32>
    %12 = tosa.maximum %8, %2 : (tensor<91x62xf32>, tensor<91x62xf32>) -> tensor<91x62xf32>
    %13 = tosa.concat %12, %10 {axis = 1 : i32} : (tensor<91x62xf32>, tensor<91x62xf32>) -> tensor<91x124xf32>
    %14 = tosa.exp %12 : (tensor<91x62xf32>) -> tensor<91x62xf32>
    %15 = tosa.pow %14, %6 : (tensor<91x62xf32>, tensor<91x62xf32>) -> tensor<91x62xf32>
    %16 = tosa.rsqrt %13 : (tensor<91x124xf32>) -> tensor<91x124xf32>
    %r_17 = tosa.const_shape {values = dense<[ 11284 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %17 = tosa.reshape %16, %r_17 : (tensor<91x124xf32>, !tosa.shape<1>) -> tensor<11284xf32>
    %18 = tosa.exp %17 : (tensor<11284xf32>) -> tensor<11284xf32>
    %19 = tosa.sigmoid %15 : (tensor<91x62xf32>) -> tensor<91x62xf32>
    %20 = tosa.logical_right_shift %11, %11 : (tensor<62xi32>, tensor<62xi32>) -> tensor<62xi32>
    return %3, %7, %9, %18, %19, %20 : tensor<1x83x1x96xi1>, tensor<44x19x93x12x57xi32>, tensor<1x83x49x96xi1>, tensor<11284xf32>, tensor<91x62xf32>, tensor<62xi32>
  }
}
