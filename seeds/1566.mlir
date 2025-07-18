module {
  func.func @main(%arg0: tensor<20x2x42xi32>, %arg1: tensor<20x2x1xi32>, %arg2: tensor<93xf32>, %arg3: tensor<64x53x52x72xf32>, %arg4: tensor<97x78x27x25xf32>, %arg5: tensor<97xf32>) -> (tensor<64x132x81x97xf32>, tensor<8xf32>, tensor<60x6x84xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<20x2x42xi32>, tensor<20x2x1xi32>) -> tensor<20x2x42xi32>
    %1 = tosa.floor %arg2 : (tensor<93xf32>) -> tensor<93xf32>
    %t_2 = tosa.const_shape {values = dense<[ 3, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %0, %t_2 : (tensor<20x2x42xi32>, !tosa.shape<3>) -> tensor<60x6x84xi32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 64, 132, 81, 97>} : (tensor<64x53x52x72xf32>, tensor<97x78x27x25xf32>, tensor<97xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<64x132x81x97xf32>
    %s_4_start = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_4_size = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.slice %1, %s_4_start, %s_4_size : (tensor<93xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8xf32>
    %5 = tosa.greater %2, %2 : (tensor<60x6x84xi32>, tensor<60x6x84xi32>) -> tensor<60x6x84xi1>
    return %3, %4, %5 : tensor<64x132x81x97xf32>, tensor<8xf32>, tensor<60x6x84xi1>
  }
}
