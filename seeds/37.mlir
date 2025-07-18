module {
  func.func @main(%arg0: tensor<93x42xf32>, %arg1: tensor<44x40x7xi1>, %arg2: tensor<34x62x51x67x50x64xi32>, %arg3: tensor<34x62x51x67x50x64xi32>) -> (tensor<34x62x51x67x50x64xi32>, tensor<93x42xf32>, tensor<93x1xi1>, tensor<2x80x1xi1>, tensor<1x80x1xi1>) {
    %0 = tosa.floor %arg0 : (tensor<93x42xf32>) -> tensor<93x42xf32>
    %1 = tosa.floor %0 : (tensor<93x42xf32>) -> tensor<93x42xf32>
    %2 = tosa.reduce_product %1 {axis = 1 : i32} : (tensor<93x42xf32>) -> tensor<93x1xf32>
    %3 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<44x40x7xi1>) -> tensor<1x40x7xi1>
    %4 = tosa.clamp %2 {min_val = -4.600000e+01 : f32, max_val = -2.500000e+01 : f32} : (tensor<93x1xf32>) -> tensor<93x1xf32>
    %5 = tosa.minimum %4, %2 : (tensor<93x1xf32>, tensor<93x1xf32>) -> tensor<93x1xf32>
    %6 = tosa.greater %5, %5 : (tensor<93x1xf32>, tensor<93x1xf32>) -> tensor<93x1xi1>
    %7 = tosa.reduce_sum %3 {axis = 2 : i32} : (tensor<1x40x7xi1>) -> tensor<1x40x1xi1>
    %8 = tosa.abs %7 : (tensor<1x40x1xi1>) -> tensor<1x40x1xi1>
    %9 = tosa.reduce_sum %8 {axis = 2 : i32} : (tensor<1x40x1xi1>) -> tensor<1x40x1xi1>
    %10 = tosa.bitwise_or %9, %8 : (tensor<1x40x1xi1>, tensor<1x40x1xi1>) -> tensor<1x40x1xi1>
    %11 = tosa.logical_right_shift %6, %6 : (tensor<93x1xi1>, tensor<93x1xi1>) -> tensor<93x1xi1>
    %12 = tosa.add %11, %6 : (tensor<93x1xi1>, tensor<93x1xi1>) -> tensor<93x1xi1>
    %13 = tosa.logical_xor %12, %6 : (tensor<93x1xi1>, tensor<93x1xi1>) -> tensor<93x1xi1>
    %14 = tosa.bitwise_and %13, %12 : (tensor<93x1xi1>, tensor<93x1xi1>) -> tensor<93x1xi1>
    %15 = tosa.intdiv %arg2, %arg3 : (tensor<34x62x51x67x50x64xi32>, tensor<34x62x51x67x50x64xi32>) -> tensor<34x62x51x67x50x64xi32>
    %16 = tosa.floor %1 : (tensor<93x42xf32>) -> tensor<93x42xf32>
    %17 = tosa.logical_not %14 : (tensor<93x1xi1>) -> tensor<93x1xi1>
    %18 = tosa.reduce_max %10 {axis = 0 : i32} : (tensor<1x40x1xi1>) -> tensor<1x40x1xi1>
    %19 = tosa.logical_right_shift %17, %13 : (tensor<93x1xi1>, tensor<93x1xi1>) -> tensor<93x1xi1>
    %20 = tosa.clz %18 : (tensor<1x40x1xi1>) -> tensor<1x40x1xi1>
    %21 = tosa.concat %20, %7 {axis = 1 : i32} : (tensor<1x40x1xi1>, tensor<1x40x1xi1>) -> tensor<1x80x1xi1>
    %22 = tosa.logical_right_shift %19, %19 : (tensor<93x1xi1>, tensor<93x1xi1>) -> tensor<93x1xi1>
    %23 = tosa.bitwise_and %21, %21 : (tensor<1x80x1xi1>, tensor<1x80x1xi1>) -> tensor<1x80x1xi1>
    %24 = tosa.concat %23, %23 {axis = 0 : i32} : (tensor<1x80x1xi1>, tensor<1x80x1xi1>) -> tensor<2x80x1xi1>
    %25 = tosa.logical_not %23 : (tensor<1x80x1xi1>) -> tensor<1x80x1xi1>
    return %15, %16, %22, %24, %25 : tensor<34x62x51x67x50x64xi32>, tensor<93x42xf32>, tensor<93x1xi1>, tensor<2x80x1xi1>, tensor<1x80x1xi1>
  }
}
