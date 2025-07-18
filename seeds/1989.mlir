module {
  func.func @main(%arg0: tensor<25x4x50x50x79x28xi64>, %arg1: tensor<1x1x50x1x1x1xi64>, %arg2: tensor<18x81x64x92xi64>) -> (tensor<18x81x64x1xi64>, tensor<10x12x8x10x5x4xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<25x4x50x50x79x28xi64>, tensor<1x1x50x1x1x1xi64>) -> tensor<25x4x50x50x79x28xi64>
    %1 = tosa.clamp %0 {min_val = -19 : i64, max_val = 30 : i64} : (tensor<25x4x50x50x79x28xi64>) -> tensor<25x4x50x50x79x28xi64>
    %2 = tosa.bitwise_not %1 : (tensor<25x4x50x50x79x28xi64>) -> tensor<25x4x50x50x79x28xi64>
    %3 = tosa.reduce_max %arg2 {axis = 3 : i32} : (tensor<18x81x64x92xi64>) -> tensor<18x81x64x1xi64>
    %4 = tosa.equal %2, %0 : (tensor<25x4x50x50x79x28xi64>, tensor<25x4x50x50x79x28xi64>) -> tensor<25x4x50x50x79x28xi1>
    %5 = tosa.identity %4 : (tensor<25x4x50x50x79x28xi1>) -> tensor<25x4x50x50x79x28xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 6, 0, 5, 13, 25, 3 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_6_size = tosa.const_shape {values = dense<[ 10, 12, 8, 10, 5, 4 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<25x4x50x50x79x28xi1>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<10x12x8x10x5x4xi1>
    return %3, %6 : tensor<18x81x64x1xi64>, tensor<10x12x8x10x5x4xi1>
  }
}
