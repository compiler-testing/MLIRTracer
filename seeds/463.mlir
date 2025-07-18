module {
  func.func @main(%arg0: tensor<24xi1>, %arg1: tensor<90x87xi32>, %arg2: tensor<1x1xi32>, %arg3: tensor<98x67x35x66x60xf32>) -> (tensor<90x87xi32>, tensor<98x67x35x66x60xf32>, tensor<1x1x1xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<24xi1>) -> tensor<1xi1>
    %1 = tosa.logical_xor %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.reduce_any %1 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.intdiv %arg1, %arg2 : (tensor<90x87xi32>, tensor<1x1xi32>) -> tensor<90x87xi32>
    %4 = tosa.rsqrt %arg3 : (tensor<98x67x35x66x60xf32>) -> tensor<98x67x35x66x60xf32>
    %r_5 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.reshape %2, %r_5 : (tensor<1xi1>, !tosa.shape<3>) -> tensor<1x1x1xi1>
    %6 = tosa.bitwise_and %5, %5 : (tensor<1x1x1xi1>, tensor<1x1x1xi1>) -> tensor<1x1x1xi1>
    return %3, %4, %6 : tensor<90x87xi32>, tensor<98x67x35x66x60xf32>, tensor<1x1x1xi1>
  }
}
