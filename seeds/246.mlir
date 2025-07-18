module {
  func.func @main(%arg0: tensor<71x27xi32>, %arg1: tensor<4x46x53x58xi1>) -> (tensor<8x6xi32>, tensor<4x1x58xi32>, tensor<4x1537x2xi1>) {
    %0 = tosa.identity %arg0 : (tensor<71x27xi32>) -> tensor<71x27xi32>
    %t_1 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<71x27xi32>, !tosa.shape<2>) -> tensor<213x54xi32>
    %2 = tosa.add %1, %1 : (tensor<213x54xi32>, tensor<213x54xi32>) -> tensor<213x54xi32>
    %3 = tosa.logical_not %arg1 : (tensor<4x46x53x58xi1>) -> tensor<4x46x53x58xi1>
    %4 = tosa.logical_left_shift %2, %1 : (tensor<213x54xi32>, tensor<213x54xi32>) -> tensor<213x54xi32>
    %5 = tosa.reverse %3 {axis = 3 : i32} : (tensor<4x46x53x58xi1>) -> tensor<4x46x53x58xi1>
    %6 = tosa.minimum %4, %1 : (tensor<213x54xi32>, tensor<213x54xi32>) -> tensor<213x54xi32>
    %7 = tosa.reduce_max %5 {axis = 1 : i32} : (tensor<4x46x53x58xi1>) -> tensor<4x1x53x58xi1>
    %8 = tosa.bitwise_not %7 : (tensor<4x1x53x58xi1>) -> tensor<4x1x53x58xi1>
    %9 = tosa.bitwise_and %6, %6 : (tensor<213x54xi32>, tensor<213x54xi32>) -> tensor<213x54xi32>
    %s_10_start = tosa.const_shape {values = dense<[ 139, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_10_size = tosa.const_shape {values = dense<[ 8, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %10 = tosa.slice %9, %s_10_start, %s_10_size : (tensor<213x54xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<8x6xi32>
    %11 = tosa.logical_xor %8, %7 : (tensor<4x1x53x58xi1>, tensor<4x1x53x58xi1>) -> tensor<4x1x53x58xi1>
    %12 = tosa.argmax %8 {axis = 2 : i32} : (tensor<4x1x53x58xi1>) -> tensor<4x1x58xi32>
    %r_13 = tosa.const_shape {values = dense<[ 4, 1537, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %13 = tosa.reshape %11, %r_13 : (tensor<4x1x53x58xi1>, !tosa.shape<3>) -> tensor<4x1537x2xi1>
    return %10, %12, %13 : tensor<8x6xi32>, tensor<4x1x58xi32>, tensor<4x1537x2xi1>
  }
}
