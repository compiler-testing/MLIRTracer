module {
  func.func @main(%arg0: tensor<36x17x99x90x64x94xi1>, %arg1: tensor<36x17x99x90x64x52xi1>, %arg2: tensor<27x56x52x18xf32>, %arg3: tensor<22x29x28x6xf32>, %arg4: tensor<22xf32>, %arg5: tensor<62x73x13x19x7xi32>, %arg6: tensor<62x73x1x19x1xi32>) -> (tensor<1782x3x9xf32>, tensor<27x1x81x22xf32>, tensor<36x17x99x90x64x146xi1>, tensor<62x73x13x19x7xi32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 5 : i32} : (tensor<36x17x99x90x64x94xi1>, tensor<36x17x99x90x64x52xi1>) -> tensor<36x17x99x90x64x146xi1>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 27, 141, 81, 22>} : (tensor<27x56x52x18xf32>, tensor<22x29x28x6xf32>, tensor<22xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<27x141x81x22xf32>
    %2 = tosa.exp %1 : (tensor<27x141x81x22xf32>) -> tensor<27x141x81x22xf32>
    %3 = tosa.reduce_min %2 {axis = 1 : i32} : (tensor<27x141x81x22xf32>) -> tensor<27x1x81x22xf32>
    %r_4 = tosa.const_shape {values = dense<[ 1782, 3, 9 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %3, %r_4 : (tensor<27x1x81x22xf32>, !tosa.shape<3>) -> tensor<1782x3x9xf32>
    %5 = tosa.sub %1, %1 : (tensor<27x141x81x22xf32>, tensor<27x141x81x22xf32>) -> tensor<27x141x81x22xf32>
    %6 = tosa.reduce_product %5 {axis = 1 : i32} : (tensor<27x141x81x22xf32>) -> tensor<27x1x81x22xf32>
    %7 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<36x17x99x90x64x146xi1>, tensor<36x17x99x90x64x146xi1>) -> tensor<36x17x99x90x64x146xi1>
    %8 = tosa.rsqrt %6 : (tensor<27x1x81x22xf32>) -> tensor<27x1x81x22xf32>
    %9 = tosa.log %8 : (tensor<27x1x81x22xf32>) -> tensor<27x1x81x22xf32>
    %10 = tosa.bitwise_not %7 : (tensor<36x17x99x90x64x146xi1>) -> tensor<36x17x99x90x64x146xi1>
    %11 = tosa.intdiv %arg5, %arg6 : (tensor<62x73x13x19x7xi32>, tensor<62x73x1x19x1xi32>) -> tensor<62x73x13x19x7xi32>
    return %4, %9, %10, %11 : tensor<1782x3x9xf32>, tensor<27x1x81x22xf32>, tensor<36x17x99x90x64x146xi1>, tensor<62x73x13x19x7xi32>
  }
}
