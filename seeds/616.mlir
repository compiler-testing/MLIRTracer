module {
  func.func @main(%arg0: tensor<21x16x49xi64>, %arg1: tensor<f32>) -> (tensor<32x98xi32>, tensor<1x2xf32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<21x16x49xi64>) -> tensor<16x49xi32>
    %1 = tosa.add %0, %0 : (tensor<16x49xi32>, tensor<16x49xi32>) -> tensor<16x49xi32>
    %2 = tosa.exp %arg1 : (tensor<f32>) -> tensor<f32>
    %t_3 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %1, %t_3 : (tensor<16x49xi32>, !tosa.shape<2>) -> tensor<32x98xi32>
    %4 = tosa.minimum %3, %3 : (tensor<32x98xi32>, tensor<32x98xi32>) -> tensor<32x98xi32>
    %r_5 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %2, %r_5 : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
    %6 = tosa.concat %5, %5 {axis = 1 : i32} : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
    return %4, %6 : tensor<32x98xi32>, tensor<1x2xf32>
  }
}
