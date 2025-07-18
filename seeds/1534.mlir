module {
  func.func @main(%arg0: tensor<13x47x47x19xi8>, %arg1: tensor<93x63x2x49x35x73xi1>, %arg2: tensor<93x63x1x1x35x1xi1>, %arg3: tensor<64x82x79x39x53xi32>, %arg4: tensor<64x82x79x1x1xi32>, %arg5: tensor<79x61xf32>) -> (tensor<13x47x47x19xi8>, tensor<93x63x2x49x35x73xi1>, tensor<64x82x79x39x53xi32>, tensor<1x3xf32>, tensor<1x102336x8374xi32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<13x47x47x19xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x47x47x19xi8>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<93x63x2x49x35x73xi1>, tensor<93x63x1x1x35x1xi1>) -> tensor<93x63x2x49x35x73xi1>
    %2 = tosa.intdiv %arg3, %arg4 : (tensor<64x82x79x39x53xi32>, tensor<64x82x79x1x1xi32>) -> tensor<64x82x79x39x53xi32>
    %3 = tosa.bitwise_or %2, %2 : (tensor<64x82x79x39x53xi32>, tensor<64x82x79x39x53xi32>) -> tensor<64x82x79x39x53xi32>
    %4 = tosa.logical_or %1, %1 : (tensor<93x63x2x49x35x73xi1>, tensor<93x63x2x49x35x73xi1>) -> tensor<93x63x2x49x35x73xi1>
    %5 = tosa.bitwise_or %2, %3 : (tensor<64x82x79x39x53xi32>, tensor<64x82x79x39x53xi32>) -> tensor<64x82x79x39x53xi32>
    %6 = tosa.floor %arg5 : (tensor<79x61xf32>) -> tensor<79x61xf32>
    %s_7_start = tosa.const_shape {values = dense<[ 9, 58 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_7_size = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.slice %6, %s_7_start, %s_7_size : (tensor<79x61xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x3xf32>
    %8 = tosa.reduce_product %7 {axis = 0 : i32} : (tensor<2x3xf32>) -> tensor<1x3xf32>
    %r_9 = tosa.const_shape {values = dense<[ 1, 102336, 8374 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.reshape %2, %r_9 : (tensor<64x82x79x39x53xi32>, !tosa.shape<3>) -> tensor<1x102336x8374xi32>
    return %0, %4, %5, %8, %9 : tensor<13x47x47x19xi8>, tensor<93x63x2x49x35x73xi1>, tensor<64x82x79x39x53xi32>, tensor<1x3xf32>, tensor<1x102336x8374xi32>
  }
}
