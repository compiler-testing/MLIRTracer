module {
  func.func @main(%arg0: tensor<61x77x60x10xi32>, %arg1: tensor<1x77x1x10xi32>, %arg2: tensor<f32>) -> (tensor<77x60x10xi32>, tensor<61x77x60x10xi32>, tensor<f32>, tensor<122x77x180x30xi32>, tensor<61x77x60x10xi32>, tensor<61x77x60x10xi32>, tensor<1x1xf32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<61x77x60x10xi32>, tensor<1x77x1x10xi32>) -> tensor<61x77x60x10xi32>
    %1 = tosa.bitwise_and %0, %0 : (tensor<61x77x60x10xi32>, tensor<61x77x60x10xi32>) -> tensor<61x77x60x10xi32>
    %2 = tosa.identity %1 : (tensor<61x77x60x10xi32>) -> tensor<61x77x60x10xi32>
    %3 = tosa.tanh %arg2 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.reciprocal %3 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.argmax %0 {axis = 0 : i32} : (tensor<61x77x60x10xi32>) -> tensor<77x60x10xi32>
    %6 = tosa.bitwise_and %1, %0 : (tensor<61x77x60x10xi32>, tensor<61x77x60x10xi32>) -> tensor<61x77x60x10xi32>
    %7 = tosa.minimum %0, %2 : (tensor<61x77x60x10xi32>, tensor<61x77x60x10xi32>) -> tensor<61x77x60x10xi32>
    %8 = tosa.exp %4 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.floor %8 : (tensor<f32>) -> tensor<f32>
    %t_10 = tosa.const_shape {values = dense<[ 2, 1, 3, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.tile %7, %t_10 : (tensor<61x77x60x10xi32>, !tosa.shape<4>) -> tensor<122x77x180x30xi32>
    %11 = tosa.clz %2 : (tensor<61x77x60x10xi32>) -> tensor<61x77x60x10xi32>
    %12 = tosa.reverse %1 {axis = 2 : i32} : (tensor<61x77x60x10xi32>) -> tensor<61x77x60x10xi32>
    %r_13 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %13 = tosa.reshape %3, %r_13 : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
    return %5, %6, %9, %10, %11, %12, %13 : tensor<77x60x10xi32>, tensor<61x77x60x10xi32>, tensor<f32>, tensor<122x77x180x30xi32>, tensor<61x77x60x10xi32>, tensor<61x77x60x10xi32>, tensor<1x1xf32>
  }
}
