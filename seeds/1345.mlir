module {
  func.func @main(%arg0: tensor<11x54x29xf32>, %arg1: tensor<29x23x70x21xf32>, %arg2: tensor<2x79x14x55xf32>, %arg3: tensor<2xf32>, %arg4: tensor<62x50x48x11x7x15xi32>, %arg5: tensor<1x1x48x11x1x15xi32>) -> (tensor<22x54x58xi1>, tensor<62x50x48x11x7x15xi32>, tensor<29x104x156x2xf32>, tensor<29x104x156x1xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<11x54x29xf32>, !tosa.shape<3>) -> tensor<22x54x58xf32>
    %1 = tosa.equal %0, %0 : (tensor<22x54x58xf32>, tensor<22x54x58xf32>) -> tensor<22x54x58xi1>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 29, 104, 156, 2>} : (tensor<29x23x70x21xf32>, tensor<2x79x14x55xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<29x104x156x2xf32>
    %3 = tosa.greater_equal %2, %2 : (tensor<29x104x156x2xf32>, tensor<29x104x156x2xf32>) -> tensor<29x104x156x2xi1>
    %4 = tosa.intdiv %arg4, %arg5 : (tensor<62x50x48x11x7x15xi32>, tensor<1x1x48x11x1x15xi32>) -> tensor<62x50x48x11x7x15xi32>
    %5 = tosa.reduce_any %3 {axis = 3 : i32} : (tensor<29x104x156x2xi1>) -> tensor<29x104x156x1xi1>
    %6 = tosa.exp %2 : (tensor<29x104x156x2xf32>) -> tensor<29x104x156x2xf32>
    %7 = tosa.reduce_all %5 {axis = 3 : i32} : (tensor<29x104x156x1xi1>) -> tensor<29x104x156x1xi1>
    return %1, %4, %6, %7 : tensor<22x54x58xi1>, tensor<62x50x48x11x7x15xi32>, tensor<29x104x156x2xf32>, tensor<29x104x156x1xi1>
  }
}
