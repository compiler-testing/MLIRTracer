module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<84x83x73x63x70x82xi32>) -> (tensor<i16>, tensor<5x3x1x2xi32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<i16>) -> tensor<i16>
    %1 = tosa.bitwise_and %0, %0 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %2 = tosa.bitwise_not %1 : (tensor<i16>) -> tensor<i16>
    %s_3_start = tosa.const_shape {values = dense<[ 76, 16, 14, 19, 6, 72 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_3_size = tosa.const_shape {values = dense<[ 8, 10, 4, 3, 12, 10 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %3 = tosa.slice %arg1, %s_3_start, %s_3_size : (tensor<84x83x73x63x70x82xi32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<8x10x4x3x12x10xi32>
    %4 = tosa.clz %3 : (tensor<8x10x4x3x12x10xi32>) -> tensor<8x10x4x3x12x10xi32>
    %r_5 = tosa.const_shape {values = dense<[ 5, 3, 3840, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.reshape %4, %r_5 : (tensor<8x10x4x3x12x10xi32>, !tosa.shape<4>) -> tensor<5x3x3840x2xi32>
    %6 = tosa.reduce_max %5 {axis = 2 : i32} : (tensor<5x3x3840x2xi32>) -> tensor<5x3x1x2xi32>
    return %2, %6 : tensor<i16>, tensor<5x3x1x2xi32>
  }
}
