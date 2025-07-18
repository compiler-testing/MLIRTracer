module {
  func.func @main(%arg0: tensor<48x51x15x90xf32>, %arg1: tensor<56x65x55x95xf32>, %arg2: tensor<56xf32>, %arg3: tensor<8xi1>, %arg4: tensor<8xi1>) -> (tensor<48x117x72x56xf32>, tensor<48x117x72x56xi1>, tensor<1x4x1xi32>, tensor<1x1xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 48, 117, 72, 56>} : (tensor<48x51x15x90xf32>, tensor<56x65x55x95xf32>, tensor<56xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<48x117x72x56xf32>
    %1 = tosa.logical_and %arg3, %arg4 : (tensor<8xi1>, tensor<8xi1>) -> tensor<8xi1>
    %r_2 = tosa.const_shape {values = dense<[ 1, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<8xi1>, !tosa.shape<2>) -> tensor<1x8xi1>
    %3 = tosa.add %0, %0 : (tensor<48x117x72x56xf32>, tensor<48x117x72x56xf32>) -> tensor<48x117x72x56xf32>
    %4 = tosa.pow %3, %0 : (tensor<48x117x72x56xf32>, tensor<48x117x72x56xf32>) -> tensor<48x117x72x56xf32>
    %5 = tosa.bitwise_and %2, %2 : (tensor<1x8xi1>, tensor<1x8xi1>) -> tensor<1x8xi1>
    %6 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %7 = tosa.transpose %5 {perms = array<i32: 0, 1>} : (tensor<1x8xi1>) -> tensor<1x8xi1>
    %8 = tosa.tanh %4 : (tensor<48x117x72x56xf32>) -> tensor<48x117x72x56xf32>
    %9 = tosa.maximum %0, %4 : (tensor<48x117x72x56xf32>, tensor<48x117x72x56xf32>) -> tensor<48x117x72x56xf32>
    %10 = tosa.reduce_all %2 {axis = 1 : i32} : (tensor<1x8xi1>) -> tensor<1x1xi1>
    %11 = tosa.greater %8, %0 : (tensor<48x117x72x56xf32>, tensor<48x117x72x56xf32>) -> tensor<48x117x72x56xi1>
    %12 = tosa.bitwise_not %7 : (tensor<1x8xi1>) -> tensor<1x8xi1>
    %r_13 = tosa.const_shape {values = dense<[ 1, 2, 4, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %13 = tosa.reshape %12, %r_13 : (tensor<1x8xi1>, !tosa.shape<4>) -> tensor<1x2x4x1xi1>
    %14 = tosa.argmax %13 {axis = 1 : i32} : (tensor<1x2x4x1xi1>) -> tensor<1x4x1xi32>
    %15 = tosa.sub %10, %10 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    return %9, %11, %14, %15 : tensor<48x117x72x56xf32>, tensor<48x117x72x56xi1>, tensor<1x4x1xi32>, tensor<1x1xi1>
  }
}
