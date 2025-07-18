module {
  func.func @main(%arg0: tensor<62x83x87x65xf32>, %arg1: tensor<15x26x24x63x36x55xi32>, %arg2: tensor<1x1x24x63x36x1xi32>, %arg3: tensor<85x53x95x71x99x49xi1>, %arg4: tensor<1x53x1x1x1x49xi1>) -> (tensor<36x63x26x24x15x55xi1>, tensor<15x26x24x63x36x55xi1>, tensor<62x83x1x1xf32>, tensor<85x53x95x71x99x49xi1>, tensor<62x83x87x1xf32>, tensor<170x53x95x71x99x49xi1>, tensor<62x83x87x1xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 3 : i32} : (tensor<62x83x87x65xf32>) -> tensor<62x83x87x1xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<15x26x24x63x36x55xi32>, tensor<1x1x24x63x36x1xi32>) -> tensor<15x26x24x63x36x55xi32>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<15x26x24x63x36x55xi32>, tensor<15x26x24x63x36x55xi32>) -> tensor<15x26x24x63x36x55xi32>
    %3 = tosa.bitwise_and %2, %1 : (tensor<15x26x24x63x36x55xi32>, tensor<15x26x24x63x36x55xi32>) -> tensor<15x26x24x63x36x55xi32>
    %4 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %5 = tosa.transpose %3 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<15x26x24x63x36x55xi32>) -> tensor<36x63x26x24x15x55xi32>
    %6 = tosa.arithmetic_right_shift %1, %2 {round = true} : (tensor<15x26x24x63x36x55xi32>, tensor<15x26x24x63x36x55xi32>) -> tensor<15x26x24x63x36x55xi32>
    %7 = tosa.intdiv %6, %1 : (tensor<15x26x24x63x36x55xi32>, tensor<15x26x24x63x36x55xi32>) -> tensor<15x26x24x63x36x55xi32>
    %8 = tosa.ceil %0 : (tensor<62x83x87x1xf32>) -> tensor<62x83x87x1xf32>
    %9 = tosa.reduce_sum %8 {axis = 3 : i32} : (tensor<62x83x87x1xf32>) -> tensor<62x83x87x1xf32>
    %10 = tosa.logical_xor %arg3, %arg4 : (tensor<85x53x95x71x99x49xi1>, tensor<1x53x1x1x1x49xi1>) -> tensor<85x53x95x71x99x49xi1>
    %11 = tosa.reduce_sum %0 {axis = 3 : i32} : (tensor<62x83x87x1xf32>) -> tensor<62x83x87x1xf32>
    %12 = tosa.log %9 : (tensor<62x83x87x1xf32>) -> tensor<62x83x87x1xf32>
    %13 = tosa.greater %5, %5 : (tensor<36x63x26x24x15x55xi32>, tensor<36x63x26x24x15x55xi32>) -> tensor<36x63x26x24x15x55xi1>
    %14 = tosa.ceil %12 : (tensor<62x83x87x1xf32>) -> tensor<62x83x87x1xf32>
    %15 = tosa.greater %7, %2 : (tensor<15x26x24x63x36x55xi32>, tensor<15x26x24x63x36x55xi32>) -> tensor<15x26x24x63x36x55xi1>
    %16 = tosa.reduce_max %9 {axis = 2 : i32} : (tensor<62x83x87x1xf32>) -> tensor<62x83x1x1xf32>
    %in_zp_17 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_17 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %17 = tosa.negate %10, %in_zp_17, %out_zp_17 : (tensor<85x53x95x71x99x49xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<85x53x95x71x99x49xi1>
    %18 = tosa.arithmetic_right_shift %17, %17 {round = true} : (tensor<85x53x95x71x99x49xi1>, tensor<85x53x95x71x99x49xi1>) -> tensor<85x53x95x71x99x49xi1>
    %19 = tosa.logical_xor %17, %18 : (tensor<85x53x95x71x99x49xi1>, tensor<85x53x95x71x99x49xi1>) -> tensor<85x53x95x71x99x49xi1>
    %20 = tosa.logical_left_shift %17, %10 : (tensor<85x53x95x71x99x49xi1>, tensor<85x53x95x71x99x49xi1>) -> tensor<85x53x95x71x99x49xi1>
    %21 = tosa.pow %12, %9 : (tensor<62x83x87x1xf32>, tensor<62x83x87x1xf32>) -> tensor<62x83x87x1xf32>
    %22 = tosa.concat %17, %20 {axis = 0 : i32} : (tensor<85x53x95x71x99x49xi1>, tensor<85x53x95x71x99x49xi1>) -> tensor<170x53x95x71x99x49xi1>
    %23 = tosa.greater %11, %14 : (tensor<62x83x87x1xf32>, tensor<62x83x87x1xf32>) -> tensor<62x83x87x1xi1>
    return %13, %15, %16, %19, %21, %22, %23 : tensor<36x63x26x24x15x55xi1>, tensor<15x26x24x63x36x55xi1>, tensor<62x83x1x1xf32>, tensor<85x53x95x71x99x49xi1>, tensor<62x83x87x1xf32>, tensor<170x53x95x71x99x49xi1>, tensor<62x83x87x1xi1>
  }
}
