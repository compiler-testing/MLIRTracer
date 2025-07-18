module {
  func.func @main(%arg0: tensor<45xi8>, %arg1: tensor<100x51xf32>) -> (tensor<9xi8>, tensor<100x51xi1>, tensor<4x12xf32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 10 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 9 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<45xi8>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<9xi8>
    %1 = tosa.floor %arg1 : (tensor<100x51xf32>) -> tensor<100x51xf32>
    %2 = tosa.identity %0 : (tensor<9xi8>) -> tensor<9xi8>
    %3 = tosa.exp %1 : (tensor<100x51xf32>) -> tensor<100x51xf32>
    %4 = tosa.reverse %3 {axis = 0 : i32} : (tensor<100x51xf32>) -> tensor<100x51xf32>
    %5 = tosa.equal %1, %1 : (tensor<100x51xf32>, tensor<100x51xf32>) -> tensor<100x51xi1>
    %6 = tosa.maximum %4, %1 : (tensor<100x51xf32>, tensor<100x51xf32>) -> tensor<100x51xf32>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %7 = tosa.negate %5, %in_zp_7, %out_zp_7 : (tensor<100x51xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<100x51xi1>
    %8 = tosa.sigmoid %6 : (tensor<100x51xf32>) -> tensor<100x51xf32>
    %9 = tosa.reverse %7 {axis = 0 : i32} : (tensor<100x51xi1>) -> tensor<100x51xi1>
    %10 = tosa.log %8 : (tensor<100x51xf32>) -> tensor<100x51xf32>
    %s_11_start = tosa.const_shape {values = dense<[ 45, 26 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_11_size = tosa.const_shape {values = dense<[ 4, 12 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %11 = tosa.slice %10, %s_11_start, %s_11_size : (tensor<100x51xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<4x12xf32>
    return %2, %9, %11 : tensor<9xi8>, tensor<100x51xi1>, tensor<4x12xf32>
  }
}
