module {
  func.func @main(%arg0: tensor<16x45xf32>) -> tensor<1x6xf32> {
    %0 = tosa.reverse %arg0 {axis = 1 : i32} : (tensor<16x45xf32>) -> tensor<16x45xf32>
    %1 = tosa.maximum %0, %0 : (tensor<16x45xf32>, tensor<16x45xf32>) -> tensor<16x45xf32>
    %r_2 = tosa.const_shape {values = dense<[ 120, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<16x45xf32>, !tosa.shape<2>) -> tensor<120x6xf32>
    %3 = tosa.reciprocal %2 : (tensor<120x6xf32>) -> tensor<120x6xf32>
    %4 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<120x6xf32>) -> tensor<1x6xf32>
    return %4 : tensor<1x6xf32>
  }
}
