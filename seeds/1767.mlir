module {
  func.func @main(%arg0: tensor<34x75xi1>, %arg1: tensor<38x7x43x65xf32>) -> (tensor<30x1x17x5xi1>, tensor<30x1x17x5xi1>, tensor<34x75xi1>, tensor<38x7x43x65xf32>, tensor<5719x1xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<34x75xi1>) -> tensor<34x75xi1>
    %r_1 = tosa.const_shape {values = dense<[ 30, 1, 17, 5 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %0, %r_1 : (tensor<34x75xi1>, !tosa.shape<4>) -> tensor<30x1x17x5xi1>
    %2 = tosa.ceil %arg1 : (tensor<38x7x43x65xf32>) -> tensor<38x7x43x65xf32>
    %3 = tosa.logical_not %1 : (tensor<30x1x17x5xi1>) -> tensor<30x1x17x5xi1>
    %4 = tosa.rsqrt %2 : (tensor<38x7x43x65xf32>) -> tensor<38x7x43x65xf32>
    %5 = tosa.logical_xor %0, %0 : (tensor<34x75xi1>, tensor<34x75xi1>) -> tensor<34x75xi1>
    %6 = tosa.floor %2 : (tensor<38x7x43x65xf32>) -> tensor<38x7x43x65xf32>
    %7 = tosa.floor %6 : (tensor<38x7x43x65xf32>) -> tensor<38x7x43x65xf32>
    %in_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %8 = tosa.negate %7, %in_zp_8, %out_zp_8 : (tensor<38x7x43x65xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<38x7x43x65xf32>
    %9 = tosa.bitwise_or %1, %1 : (tensor<30x1x17x5xi1>, tensor<30x1x17x5xi1>) -> tensor<30x1x17x5xi1>
    %r_10 = tosa.const_shape {values = dense<[ 5719, 130 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %10 = tosa.reshape %4, %r_10 : (tensor<38x7x43x65xf32>, !tosa.shape<2>) -> tensor<5719x130xf32>
    %11 = tosa.logical_not %9 : (tensor<30x1x17x5xi1>) -> tensor<30x1x17x5xi1>
    %12 = tosa.reduce_product %10 {axis = 1 : i32} : (tensor<5719x130xf32>) -> tensor<5719x1xf32>
    %13 = tosa.pow %8, %6 : (tensor<38x7x43x65xf32>, tensor<38x7x43x65xf32>) -> tensor<38x7x43x65xf32>
    %14 = tosa.bitwise_and %5, %0 : (tensor<34x75xi1>, tensor<34x75xi1>) -> tensor<34x75xi1>
    %15 = tosa.clz %11 : (tensor<30x1x17x5xi1>) -> tensor<30x1x17x5xi1>
    %in_zp_16 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_16 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %16 = tosa.negate %13, %in_zp_16, %out_zp_16 : (tensor<38x7x43x65xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<38x7x43x65xf32>
    %17 = tosa.tanh %12 : (tensor<5719x1xf32>) -> tensor<5719x1xf32>
    %18 = tosa.minimum %16, %4 : (tensor<38x7x43x65xf32>, tensor<38x7x43x65xf32>) -> tensor<38x7x43x65xf32>
    %19 = tosa.bitwise_not %14 : (tensor<34x75xi1>) -> tensor<34x75xi1>
    %20 = tosa.rsqrt %18 : (tensor<38x7x43x65xf32>) -> tensor<38x7x43x65xf32>
    %21 = tosa.reduce_product %17 {axis = 1 : i32} : (tensor<5719x1xf32>) -> tensor<5719x1xf32>
    %22 = tosa.reduce_sum %21 {axis = 1 : i32} : (tensor<5719x1xf32>) -> tensor<5719x1xf32>
    return %3, %15, %19, %20, %22 : tensor<30x1x17x5xi1>, tensor<30x1x17x5xi1>, tensor<34x75xi1>, tensor<38x7x43x65xf32>, tensor<5719x1xf32>
  }
}
