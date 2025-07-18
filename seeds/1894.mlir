module {
  func.func @main(%arg0: tensor<25x17x55xi16>, %arg1: tensor<1x97x58x1x98xi8>, %arg2: tensor<1x1x1x1x98xi8>, %arg3: tensor<25xf32>) -> (tensor<25x17x1xi16>, tensor<25xf32>, tensor<1x97x58x1x98xi1>, tensor<3xf32>, tensor<25xi1>, tensor<1xf32>) {
    %0 = tosa.reduce_min %arg0 {axis = 2 : i32} : (tensor<25x17x55xi16>) -> tensor<25x17x1xi16>
    %1 = tosa.greater_equal %arg1, %arg2 : (tensor<1x97x58x1x98xi8>, tensor<1x1x1x1x98xi8>) -> tensor<1x97x58x1x98xi1>
    %2 = tosa.floor %arg3 : (tensor<25xf32>) -> tensor<25xf32>
    %3 = tosa.logical_left_shift %0, %0 : (tensor<25x17x1xi16>, tensor<25x17x1xi16>) -> tensor<25x17x1xi16>
    %4 = tosa.rsqrt %2 : (tensor<25xf32>) -> tensor<25xf32>
    %5 = tosa.bitwise_xor %1, %1 : (tensor<1x97x58x1x98xi1>, tensor<1x97x58x1x98xi1>) -> tensor<1x97x58x1x98xi1>
    %6 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %7 = tosa.transpose %2 {perms = array<i32: 0>} : (tensor<25xf32>) -> tensor<25xf32>
    %8 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<25xf32>) -> tensor<1xf32>
    %9 = tosa.greater %2, %7 : (tensor<25xf32>, tensor<25xf32>) -> tensor<25xi1>
    %10 = tosa.exp %8 : (tensor<1xf32>) -> tensor<1xf32>
    %11 = tosa.reduce_min %10 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %s_12_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_12_size = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %12 = tosa.slice %11, %s_12_start, %s_12_size : (tensor<1xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
    %13 = tosa.logical_left_shift %9, %9 : (tensor<25xi1>, tensor<25xi1>) -> tensor<25xi1>
    %14 = tosa.clz %13 : (tensor<25xi1>) -> tensor<25xi1>
    %15 = tosa.sub %9, %9 : (tensor<25xi1>, tensor<25xi1>) -> tensor<25xi1>
    %16 = tosa.clz %15 : (tensor<25xi1>) -> tensor<25xi1>
    %17 = tosa.sub %16, %14 : (tensor<25xi1>, tensor<25xi1>) -> tensor<25xi1>
    %18 = tosa.minimum %8, %10 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %3, %4, %5, %12, %17, %18 : tensor<25x17x1xi16>, tensor<25xf32>, tensor<1x97x58x1x98xi1>, tensor<3xf32>, tensor<25xi1>, tensor<1xf32>
  }
}
