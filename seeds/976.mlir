module {
  func.func @main(%arg0: tensor<15x39x29xf32>, %arg1: tensor<3x2xi32>, %arg2: tensor<24xi1>, %arg3: tensor<1xi1>) -> (tensor<15x39x29xf32>, tensor<24xi1>, tensor<1xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<15x39x29xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<15x39x29xf32>
    %1 = tosa.rsqrt %0 : (tensor<15x39x29xf32>) -> tensor<15x39x29xf32>
    %2 = tosa.ceil %1 : (tensor<15x39x29xf32>) -> tensor<15x39x29xf32>
    %3 = tosa.reciprocal %2 : (tensor<15x39x29xf32>) -> tensor<15x39x29xf32>
    %4 = tosa.logical_left_shift %arg2, %arg3 : (tensor<24xi1>, tensor<1xi1>) -> tensor<24xi1>
    %5 = tosa.reverse %3 {axis = 0 : i32} : (tensor<15x39x29xf32>) -> tensor<15x39x29xf32>
    %6 = tosa.reduce_all %4 {axis = 0 : i32} : (tensor<24xi1>) -> tensor<1xi1>
    %7 = tosa.logical_or %4, %4 : (tensor<24xi1>, tensor<24xi1>) -> tensor<24xi1>
    %8 = tosa.bitwise_and %6, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %9 = tosa.logical_not %8 : (tensor<1xi1>) -> tensor<1xi1>
    return %5, %7, %9 : tensor<15x39x29xf32>, tensor<24xi1>, tensor<1xi1>
  }
}
