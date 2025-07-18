module {
  func.func @main(%arg0: tensor<56x68x20x30xf32>, %arg1: tensor<22x34x75x45xf32>, %arg2: tensor<22xf32>) -> (tensor<56x104x98x22xi1>, tensor<3x4x18xi32>, tensor<56x104x98x22xf32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 56, 104, 98, 22>} : (tensor<56x68x20x30xf32>, tensor<22x34x75x45xf32>, tensor<22xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<56x104x98x22xf32>
    %1 = tosa.equal %0, %0 : (tensor<56x104x98x22xf32>, tensor<56x104x98x22xf32>) -> tensor<56x104x98x22xi1>
    %2 = tosa.bitwise_xor %1, %1 : (tensor<56x104x98x22xi1>, tensor<56x104x98x22xi1>) -> tensor<56x104x98x22xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<56x104x98x22xi1>, tensor<56x104x98x22xi1>) -> tensor<56x104x98x22xi1>
    %4 = tosa.clz %3 : (tensor<56x104x98x22xi1>) -> tensor<56x104x98x22xi1>
    %5 = tosa.abs %4 : (tensor<56x104x98x22xi1>) -> tensor<56x104x98x22xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 5, 20, 32, 6 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_6_size = tosa.const_shape {values = dense<[ 3, 5, 2, 9 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<56x104x98x22xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x5x2x9xi1>
    %t_7 = tosa.const_shape {values = dense<[ 1, 1, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = tosa.tile %6, %t_7 : (tensor<3x5x2x9xi1>, !tosa.shape<4>) -> tensor<3x5x4x18xi1>
    %8 = tosa.greater_equal %0, %0 : (tensor<56x104x98x22xf32>, tensor<56x104x98x22xf32>) -> tensor<56x104x98x22xi1>
    %9 = tosa.logical_xor %7, %7 : (tensor<3x5x4x18xi1>, tensor<3x5x4x18xi1>) -> tensor<3x5x4x18xi1>
    %10 = tosa.argmax %9 {axis = 1 : i32} : (tensor<3x5x4x18xi1>) -> tensor<3x4x18xi32>
    %11 = tosa.clz %10 : (tensor<3x4x18xi32>) -> tensor<3x4x18xi32>
    %12 = tosa.exp %0 : (tensor<56x104x98x22xf32>) -> tensor<56x104x98x22xf32>
    return %8, %11, %12 : tensor<56x104x98x22xi1>, tensor<3x4x18xi32>, tensor<56x104x98x22xf32>
  }
}
