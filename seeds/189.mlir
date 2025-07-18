module {
  func.func @main(%arg0: tensor<80xi1>, %arg1: tensor<20x32x92xf32>, %arg2: tensor<20x1x1xf32>, %arg3: tensor<98x34x94x21xf32>, %arg4: tensor<59x99x33x36xf32>, %arg5: tensor<59xf32>, %arg6: tensor<84x95x12x23xi32>, %arg7: tensor<1x1x12x1xi32>) -> (tensor<240xi1>, tensor<98x135x130x59xf32>, tensor<20x32x92xf32>, tensor<84x95x12x23xi32>, tensor<20x96x276xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<80xi1>, !tosa.shape<1>) -> tensor<240xi1>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<20x32x92xf32>, tensor<20x1x1xf32>) -> tensor<20x32x92xf32>
    %2 = tosa.bitwise_not %0 : (tensor<240xi1>) -> tensor<240xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 98, 135, 130, 59>} : (tensor<98x34x94x21xf32>, tensor<59x99x33x36xf32>, tensor<59xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<98x135x130x59xf32>
    %4 = tosa.ceil %1 : (tensor<20x32x92xf32>) -> tensor<20x32x92xf32>
    %5 = tosa.add %4, %1 : (tensor<20x32x92xf32>, tensor<20x32x92xf32>) -> tensor<20x32x92xf32>
    %6 = tosa.minimum %4, %1 : (tensor<20x32x92xf32>, tensor<20x32x92xf32>) -> tensor<20x32x92xf32>
    %7 = tosa.rsqrt %5 : (tensor<20x32x92xf32>) -> tensor<20x32x92xf32>
    %8 = tosa.clamp %7 {min_val = -3.100000e+01 : f32, max_val = 4.600000e+01 : f32} : (tensor<20x32x92xf32>) -> tensor<20x32x92xf32>
    %9 = tosa.log %8 : (tensor<20x32x92xf32>) -> tensor<20x32x92xf32>
    %10 = tosa.ceil %9 : (tensor<20x32x92xf32>) -> tensor<20x32x92xf32>
    %t_11 = tosa.const_shape {values = dense<[ 1, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %11 = tosa.tile %10, %t_11 : (tensor<20x32x92xf32>, !tosa.shape<3>) -> tensor<20x96x276xf32>
    %12 = tosa.intdiv %arg6, %arg7 : (tensor<84x95x12x23xi32>, tensor<1x1x12x1xi32>) -> tensor<84x95x12x23xi32>
    %13 = tosa.add %12, %12 : (tensor<84x95x12x23xi32>, tensor<84x95x12x23xi32>) -> tensor<84x95x12x23xi32>
    %14 = tosa.abs %11 : (tensor<20x96x276xf32>) -> tensor<20x96x276xf32>
    return %2, %3, %6, %13, %14 : tensor<240xi1>, tensor<98x135x130x59xf32>, tensor<20x32x92xf32>, tensor<84x95x12x23xi32>, tensor<20x96x276xf32>
  }
}
