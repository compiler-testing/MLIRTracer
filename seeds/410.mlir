module {
  func.func @main(%arg0: tensor<27x82x66x22xi16>, %arg1: tensor<52x66x71xi1>, %arg2: tensor<63x5x51x16xf32>, %arg3: tensor<24x63x61x37xf32>, %arg4: tensor<24xf32>) -> (tensor<3214728xi16>, tensor<63x70x114x24xi1>, tensor<1x1x71xi1>, tensor<63x70x114x24xf32>, tensor<63x70x114x24xf32>) {
    %r_0 = tosa.const_shape {values = dense<[ 3214728 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<27x82x66x22xi16>, !tosa.shape<1>) -> tensor<3214728xi16>
    %1 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<52x66x71xi1>) -> tensor<52x1x71xi1>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 63, 70, 114, 24>} : (tensor<63x5x51x16xf32>, tensor<24x63x61x37xf32>, tensor<24xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<63x70x114x24xf32>
    %3 = "tosa.const"() {values = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
    %4 = tosa.transpose %1 {perms = array<i32: 0, 1, 2>} : (tensor<52x1x71xi1>) -> tensor<52x1x71xi1>
    %5 = tosa.greater_equal %2, %2 : (tensor<63x70x114x24xf32>, tensor<63x70x114x24xf32>) -> tensor<63x70x114x24xi1>
    %6 = tosa.reduce_all %4 {axis = 0 : i32} : (tensor<52x1x71xi1>) -> tensor<1x1x71xi1>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<1x1x71xi1>, tensor<1x1x71xi1>) -> tensor<1x1x71xi1>
    %8 = tosa.floor %2 : (tensor<63x70x114x24xf32>) -> tensor<63x70x114x24xf32>
    %9 = tosa.maximum %2, %2 : (tensor<63x70x114x24xf32>, tensor<63x70x114x24xf32>) -> tensor<63x70x114x24xf32>
    return %0, %5, %7, %8, %9 : tensor<3214728xi16>, tensor<63x70x114x24xi1>, tensor<1x1x71xi1>, tensor<63x70x114x24xf32>, tensor<63x70x114x24xf32>
  }
}
