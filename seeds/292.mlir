module {
  func.func @main(%arg0: tensor<66xf32>, %arg1: tensor<66xf32>, %arg2: tensor<24x64xf32>, %arg3: tensor<42x66x13x46x49xi32>, %arg4: tensor<42x66x13x46x49xi32>) -> (tensor<24x64xi1>, tensor<42x66x13x46x49xi32>, tensor<48xi32>, tensor<42x66x13x46x49xi32>, tensor<66xi1>, tensor<256x24xf32>, tensor<24x64xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<66xf32>, tensor<66xf32>) -> tensor<66xi1>
    %1 = tosa.sigmoid %arg2 : (tensor<24x64xf32>) -> tensor<24x64xf32>
    %2 = tosa.tanh %1 : (tensor<24x64xf32>) -> tensor<24x64xf32>
    %t_3 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %1, %t_3 : (tensor<24x64xf32>, !tosa.shape<2>) -> tensor<48x128xf32>
    %4 = tosa.ceil %3 : (tensor<48x128xf32>) -> tensor<48x128xf32>
    %5 = tosa.logical_not %0 : (tensor<66xi1>) -> tensor<66xi1>
    %6 = tosa.equal %2, %1 : (tensor<24x64xf32>, tensor<24x64xf32>) -> tensor<24x64xi1>
    %7 = tosa.maximum %4, %3 : (tensor<48x128xf32>, tensor<48x128xf32>) -> tensor<48x128xf32>
    %8 = tosa.tanh %7 : (tensor<48x128xf32>) -> tensor<48x128xf32>
    %9 = tosa.bitwise_and %5, %0 : (tensor<66xi1>, tensor<66xi1>) -> tensor<66xi1>
    %10 = tosa.intdiv %arg3, %arg4 : (tensor<42x66x13x46x49xi32>, tensor<42x66x13x46x49xi32>) -> tensor<42x66x13x46x49xi32>
    %r_11 = tosa.const_shape {values = dense<[ 256, 24 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %11 = tosa.reshape %8, %r_11 : (tensor<48x128xf32>, !tosa.shape<2>) -> tensor<256x24xf32>
    %12 = tosa.sigmoid %11 : (tensor<256x24xf32>) -> tensor<256x24xf32>
    %13 = tosa.intdiv %10, %10 : (tensor<42x66x13x46x49xi32>, tensor<42x66x13x46x49xi32>) -> tensor<42x66x13x46x49xi32>
    %14 = tosa.argmax %4 {axis = 1 : i32} : (tensor<48x128xf32>) -> tensor<48xi32>
    %15 = tosa.abs %10 : (tensor<42x66x13x46x49xi32>) -> tensor<42x66x13x46x49xi32>
    %16 = tosa.logical_and %9, %0 : (tensor<66xi1>, tensor<66xi1>) -> tensor<66xi1>
    %17 = tosa.reciprocal %12 : (tensor<256x24xf32>) -> tensor<256x24xf32>
    %18 = tosa.floor %1 : (tensor<24x64xf32>) -> tensor<24x64xf32>
    return %6, %13, %14, %15, %16, %17, %18 : tensor<24x64xi1>, tensor<42x66x13x46x49xi32>, tensor<48xi32>, tensor<42x66x13x46x49xi32>, tensor<66xi1>, tensor<256x24xf32>, tensor<24x64xf32>
  }
}
