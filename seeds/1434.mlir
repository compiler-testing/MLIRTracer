module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<77x60x15x63xi1>, %arg2: tensor<77x1x15x63xi1>) -> (tensor<1x1x1xi32>, tensor<f32>, tensor<i1>, tensor<4365900xi1>, tensor<i1>, tensor<f32>, tensor<f32>, tensor<i1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<77x60x15x63xi1>, tensor<77x1x15x63xi1>) -> tensor<77x60x15x63xi1>
    %2 = tosa.log %0 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.reduce_max %1 {axis = 2 : i32} : (tensor<77x60x15x63xi1>) -> tensor<77x60x1x63xi1>
    %4 = tosa.reduce_any %3 {axis = 1 : i32} : (tensor<77x60x1x63xi1>) -> tensor<77x1x1x63xi1>
    %5 = tosa.reduce_product %4 {axis = 3 : i32} : (tensor<77x1x1x63xi1>) -> tensor<77x1x1x1xi1>
    %6 = tosa.sigmoid %0 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.argmax %5 {axis = 0 : i32} : (tensor<77x1x1x1xi1>) -> tensor<1x1x1xi32>
    %8 = tosa.logical_or %1, %1 : (tensor<77x60x15x63xi1>, tensor<77x60x15x63xi1>) -> tensor<77x60x15x63xi1>
    %9 = tosa.rsqrt %0 : (tensor<f32>) -> tensor<f32>
    %10 = tosa.rsqrt %2 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.greater_equal %9, %6 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = tosa.tanh %2 : (tensor<f32>) -> tensor<f32>
    %13 = tosa.sub %1, %8 : (tensor<77x60x15x63xi1>, tensor<77x60x15x63xi1>) -> tensor<77x60x15x63xi1>
    %r_14 = tosa.const_shape {values = dense<[ 4365900 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %14 = tosa.reshape %13, %r_14 : (tensor<77x60x15x63xi1>, !tosa.shape<1>) -> tensor<4365900xi1>
    %15 = tosa.greater %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = tosa.pow %9, %12 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %17 = tosa.rsqrt %9 : (tensor<f32>) -> tensor<f32>
    %18 = tosa.greater %6, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %7, %10, %11, %14, %15, %16, %17, %18 : tensor<1x1x1xi32>, tensor<f32>, tensor<i1>, tensor<4365900xi1>, tensor<i1>, tensor<f32>, tensor<f32>, tensor<i1>
  }
}
