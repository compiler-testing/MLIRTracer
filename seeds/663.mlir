module {
  func.func @main(%arg0: tensor<23xi8>, %arg1: tensor<78x84x23x54x13x4xi1>, %arg2: tensor<78x1x23x54x13x1xi1>, %arg3: tensor<73x96x18x14x4xf32>, %arg4: tensor<1x1x18x14x4xf32>) -> (tensor<1xi1>, tensor<5880xi1>, tensor<73x96x18x14x4xf32>, tensor<73x96x18x14x4xf32>, tensor<3x1177344x2x1xf32>, tensor<78x84x23x54x26x4xi1>, tensor<73x96x18x14x4xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<23xi8>) -> tensor<1xi8>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<1xi8>) -> tensor<1xi8>
    %2 = tosa.logical_or %arg1, %arg2 : (tensor<78x84x23x54x13x4xi1>, tensor<78x1x23x54x13x1xi1>) -> tensor<78x84x23x54x13x4xi1>
    %3 = tosa.logical_xor %2, %2 : (tensor<78x84x23x54x13x4xi1>, tensor<78x84x23x54x13x4xi1>) -> tensor<78x84x23x54x13x4xi1>
    %s_4_start = tosa.const_shape {values = dense<[ 31, 64, 18, 13, 1, 0 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_4_size = tosa.const_shape {values = dense<[ 2, 7, 5, 1, 12, 7 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<78x84x23x54x13x4xi1>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<2x7x5x1x12x7xi1>
    %5 = tosa.equal %1, %0 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi1>
    %r_6 = tosa.const_shape {values = dense<[ 5880 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.reshape %4, %r_6 : (tensor<2x7x5x1x12x7xi1>, !tosa.shape<1>) -> tensor<5880xi1>
    %7 = tosa.pow %arg3, %arg4 : (tensor<73x96x18x14x4xf32>, tensor<1x1x18x14x4xf32>) -> tensor<73x96x18x14x4xf32>
    %8 = tosa.sigmoid %7 : (tensor<73x96x18x14x4xf32>) -> tensor<73x96x18x14x4xf32>
    %9 = tosa.logical_right_shift %2, %3 : (tensor<78x84x23x54x13x4xi1>, tensor<78x84x23x54x13x4xi1>) -> tensor<78x84x23x54x13x4xi1>
    %10 = tosa.abs %7 : (tensor<73x96x18x14x4xf32>) -> tensor<73x96x18x14x4xf32>
    %11 = tosa.bitwise_and %9, %2 : (tensor<78x84x23x54x13x4xi1>, tensor<78x84x23x54x13x4xi1>) -> tensor<78x84x23x54x13x4xi1>
    %12 = tosa.tanh %10 : (tensor<73x96x18x14x4xf32>) -> tensor<73x96x18x14x4xf32>
    %13 = tosa.logical_not %11 : (tensor<78x84x23x54x13x4xi1>) -> tensor<78x84x23x54x13x4xi1>
    %r_14 = tosa.const_shape {values = dense<[ 3, 1177344, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %14 = tosa.reshape %7, %r_14 : (tensor<73x96x18x14x4xf32>, !tosa.shape<4>) -> tensor<3x1177344x2x1xf32>
    %15 = tosa.bitwise_and %13, %9 : (tensor<78x84x23x54x13x4xi1>, tensor<78x84x23x54x13x4xi1>) -> tensor<78x84x23x54x13x4xi1>
    %16 = tosa.bitwise_and %15, %2 : (tensor<78x84x23x54x13x4xi1>, tensor<78x84x23x54x13x4xi1>) -> tensor<78x84x23x54x13x4xi1>
    %17 = tosa.exp %14 : (tensor<3x1177344x2x1xf32>) -> tensor<3x1177344x2x1xf32>
    %18 = tosa.minimum %17, %17 : (tensor<3x1177344x2x1xf32>, tensor<3x1177344x2x1xf32>) -> tensor<3x1177344x2x1xf32>
    %19 = tosa.concat %16, %2 {axis = 4 : i32} : (tensor<78x84x23x54x13x4xi1>, tensor<78x84x23x54x13x4xi1>) -> tensor<78x84x23x54x26x4xi1>
    %20 = tosa.clz %19 : (tensor<78x84x23x54x26x4xi1>) -> tensor<78x84x23x54x26x4xi1>
    %21 = tosa.exp %7 : (tensor<73x96x18x14x4xf32>) -> tensor<73x96x18x14x4xf32>
    return %5, %6, %8, %12, %18, %20, %21 : tensor<1xi1>, tensor<5880xi1>, tensor<73x96x18x14x4xf32>, tensor<73x96x18x14x4xf32>, tensor<3x1177344x2x1xf32>, tensor<78x84x23x54x26x4xi1>, tensor<73x96x18x14x4xf32>
  }
}
