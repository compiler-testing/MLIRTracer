module {
  func.func @main(%arg0: tensor<24x81x65xi8>, %arg1: tensor<93x87xi1>, %arg2: tensor<50x47x87x94xf32>, %arg3: tensor<81x79x27x59xf32>, %arg4: tensor<81xf32>) -> (tensor<24x81x65xi8>, tensor<50x129x116x81xf32>, tensor<186x3xi1>, tensor<93x1xi1>) {
    %0 = tosa.clz %arg0 : (tensor<24x81x65xi8>) -> tensor<24x81x65xi8>
    %1 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<93x87xi1>) -> tensor<93x1xi1>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 50, 129, 116, 81>} : (tensor<50x47x87x94xf32>, tensor<81x79x27x59xf32>, tensor<81xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<50x129x116x81xf32>
    %3 = tosa.reciprocal %2 : (tensor<50x129x116x81xf32>) -> tensor<50x129x116x81xf32>
    %t_4 = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.tile %1, %t_4 : (tensor<93x1xi1>, !tosa.shape<2>) -> tensor<186x3xi1>
    %5 = tosa.logical_and %1, %1 : (tensor<93x1xi1>, tensor<93x1xi1>) -> tensor<93x1xi1>
    return %0, %3, %4, %5 : tensor<24x81x65xi8>, tensor<50x129x116x81xf32>, tensor<186x3xi1>, tensor<93x1xi1>
  }
}
