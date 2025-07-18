module {
  func.func @main(%arg0: tensor<53x46x3x44x43xf32>, %arg1: tensor<51x7xi16>) -> (tensor<44x43x53x3x46xf32>, tensor<153xi1>, tensor<5xi32>) {
    %0 = tosa.log %arg0 : (tensor<53x46x3x44x43xf32>) -> tensor<53x46x3x44x43xf32>
    %1 = tosa.clamp %0 {min_val = -1.200000e+01 : f32, max_val = 2.400000e+01 : f32} : (tensor<53x46x3x44x43xf32>) -> tensor<53x46x3x44x43xf32>
    %2 = "tosa.const"() {values = dense<[3, 4, 0, 2, 1]> : tensor<5xi32>} : () -> tensor<5xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 3, 4, 0, 2, 1>} : (tensor<53x46x3x44x43xf32>) -> tensor<44x43x53x3x46xf32>
    %t_4 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.tile %arg1, %t_4 : (tensor<51x7xi16>, !tosa.shape<2>) -> tensor<153x7xi16>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<153x7xi16>, tensor<153x7xi16>) -> tensor<153x7xi16>
    %6 = tosa.concat %5, %5 {axis = 1 : i32} : (tensor<153x7xi16>, tensor<153x7xi16>) -> tensor<153x14xi16>
    %7 = tosa.argmax %6 {axis = 1 : i32} : (tensor<153x14xi16>) -> tensor<153xi32>
    %8 = tosa.clz %7 : (tensor<153xi32>) -> tensor<153xi32>
    %9 = tosa.add %4, %4 : (tensor<153x7xi16>, tensor<153x7xi16>) -> tensor<153x7xi16>
    %s_10_start = tosa.const_shape {values = dense<[ 9, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_10_size = tosa.const_shape {values = dense<[ 6, 5 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %10 = tosa.slice %9, %s_10_start, %s_10_size : (tensor<153x7xi16>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<6x5xi16>
    %11 = tosa.minimum %8, %8 : (tensor<153xi32>, tensor<153xi32>) -> tensor<153xi32>
    %12 = tosa.bitwise_and %11, %8 : (tensor<153xi32>, tensor<153xi32>) -> tensor<153xi32>
    %13 = tosa.equal %12, %11 : (tensor<153xi32>, tensor<153xi32>) -> tensor<153xi1>
    %14 = tosa.logical_and %13, %13 : (tensor<153xi1>, tensor<153xi1>) -> tensor<153xi1>
    %15 = tosa.logical_left_shift %14, %14 : (tensor<153xi1>, tensor<153xi1>) -> tensor<153xi1>
    %16 = tosa.argmax %10 {axis = 0 : i32} : (tensor<6x5xi16>) -> tensor<5xi32>
    %17 = tosa.clz %16 : (tensor<5xi32>) -> tensor<5xi32>
    return %3, %15, %17 : tensor<44x43x53x3x46xf32>, tensor<153xi1>, tensor<5xi32>
  }
}
