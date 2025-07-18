module {
  func.func @main(%arg0: tensor<46xi32>, %arg1: tensor<20x76x25x99xf32>, %arg2: tensor<51x70x68x63xf32>, %arg3: tensor<51xf32>, %arg4: tensor<45x76x56x36x2x44xi1>, %arg5: tensor<45x1x1x1x2x44xi1>) -> (tensor<45x76x56x36x2x44xi1>, tensor<i32>, tensor<147x95x51xi32>) {
    %t_0 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<46xi32>, !tosa.shape<1>) -> tensor<92xi32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 20, 147, 95, 51>} : (tensor<20x76x25x99xf32>, tensor<51x70x68x63xf32>, tensor<51xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<20x147x95x51xf32>
    %2 = tosa.logical_and %arg4, %arg5 : (tensor<45x76x56x36x2x44xi1>, tensor<45x1x1x1x2x44xi1>) -> tensor<45x76x56x36x2x44xi1>
    %3 = tosa.argmax %0 {axis = 0 : i32} : (tensor<92xi32>) -> tensor<i32>
    %4 = tosa.reverse %1 {axis = 2 : i32} : (tensor<20x147x95x51xf32>) -> tensor<20x147x95x51xf32>
    %5 = tosa.argmax %4 {axis = 0 : i32} : (tensor<20x147x95x51xf32>) -> tensor<147x95x51xi32>
    return %2, %3, %5 : tensor<45x76x56x36x2x44xi1>, tensor<i32>, tensor<147x95x51xi32>
  }
}
