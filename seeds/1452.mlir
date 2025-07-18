module {
  func.func @main(%arg0: tensor<63x46x78x22xf32>, %arg1: tensor<63x41x74x2xf32>, %arg2: tensor<63xf32>, %arg3: tensor<6x16xi1>, %arg4: tensor<6x16xi1>) -> (tensor<2x6x2x12xf32>, tensor<6x1xi1>, tensor<5x7xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 63, 135, 154, 63>} : (tensor<63x46x78x22xf32>, tensor<63x41x74x2xf32>, tensor<63xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<63x135x154x63xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 29, 16, 30, 51 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_1_size = tosa.const_shape {values = dense<[ 2, 6, 2, 12 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<63x135x154x63xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x6x2x12xf32>
    %2 = tosa.arithmetic_right_shift %arg3, %arg4 {round = false} : (tensor<6x16xi1>, tensor<6x16xi1>) -> tensor<6x16xi1>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<6x16xi1>, tensor<6x16xi1>) -> tensor<6x16xi1>
    %4 = tosa.logical_not %2 : (tensor<6x16xi1>) -> tensor<6x16xi1>
    %5 = tosa.abs %3 : (tensor<6x16xi1>) -> tensor<6x16xi1>
    %6 = tosa.reverse %5 {axis = 1 : i32} : (tensor<6x16xi1>) -> tensor<6x16xi1>
    %7 = tosa.reduce_product %4 {axis = 1 : i32} : (tensor<6x16xi1>) -> tensor<6x1xi1>
    %8 = tosa.add %7, %7 : (tensor<6x1xi1>, tensor<6x1xi1>) -> tensor<6x1xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %9 = tosa.negate %8, %in_zp_9, %out_zp_9 : (tensor<6x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<6x1xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 1, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_10_size = tosa.const_shape {values = dense<[ 5, 7 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %10 = tosa.slice %6, %s_10_start, %s_10_size : (tensor<6x16xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<5x7xi1>
    return %1, %9, %10 : tensor<2x6x2x12xf32>, tensor<6x1xi1>, tensor<5x7xi1>
  }
}
