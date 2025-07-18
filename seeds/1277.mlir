module {
  func.func @main(%arg0: tensor<81x27x95xi16>, %arg1: tensor<3x2xi32>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<11x99x24x13x66xi32>, %arg5: tensor<11x99x1x1x1xi32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<29x17x14xi1>) -> (tensor<i1>, tensor<11x99x24x26x66xi1>, tensor<81x27x1xi16>, tensor<1x95x27xi16>, tensor<i1>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<58xi1>, tensor<29x1x14xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<81x27x95xi16>, !tosa.shape<6>, tensor<1xi16>) -> tensor<81x27x95xi16>
    %1 = tosa.abs %0 : (tensor<81x27x95xi16>) -> tensor<81x27x95xi16>
    %2 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 0, 2, 1>} : (tensor<81x27x95xi16>) -> tensor<81x95x27xi16>
    %4 = tosa.logical_or %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.greater %arg4, %arg5 : (tensor<11x99x24x13x66xi32>, tensor<11x99x1x1x1xi32>) -> tensor<11x99x24x13x66xi1>
    %6 = tosa.pow %arg6, %arg7 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = tosa.greater_equal %6, %6 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = tosa.logical_not %7 : (tensor<i1>) -> tensor<i1>
    %9 = tosa.equal %6, %6 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %10 = tosa.concat %5, %5 {axis = 3 : i32} : (tensor<11x99x24x13x66xi1>, tensor<11x99x24x13x66xi1>) -> tensor<11x99x24x26x66xi1>
    %11 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<81x95x27xi16>) -> tensor<1x95x27xi16>
    %12 = tosa.reduce_product %0 {axis = 2 : i32} : (tensor<81x27x95xi16>) -> tensor<81x27x1xi16>
    %13 = tosa.logical_not %4 : (tensor<i1>) -> tensor<i1>
    %14 = tosa.reduce_min %11 {axis = 0 : i32} : (tensor<1x95x27xi16>) -> tensor<1x95x27xi16>
    %15 = tosa.bitwise_xor %13, %9 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %16 = tosa.reduce_all %arg8 {axis = 1 : i32} : (tensor<29x17x14xi1>) -> tensor<29x1x14xi1>
    %17 = tosa.floor %6 : (tensor<f32>) -> tensor<f32>
    %18 = tosa.tanh %6 : (tensor<f32>) -> tensor<f32>
    %19 = tosa.reduce_max %16 {axis = 2 : i32} : (tensor<29x1x14xi1>) -> tensor<29x1x1xi1>
    %t_20 = tosa.const_shape {values = dense<[ 2, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %20 = tosa.tile %19, %t_20 : (tensor<29x1x1xi1>, !tosa.shape<3>) -> tensor<58x3x1xi1>
    %21 = tosa.exp %6 : (tensor<f32>) -> tensor<f32>
    %22 = tosa.reduce_all %20 {axis = 1 : i32} : (tensor<58x3x1xi1>) -> tensor<58x1x1xi1>
    %r_23 = tosa.const_shape {values = dense<[ 58 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %23 = tosa.reshape %22, %r_23 : (tensor<58x1x1xi1>, !tosa.shape<1>) -> tensor<58xi1>
    %24 = tosa.logical_left_shift %16, %16 : (tensor<29x1x14xi1>, tensor<29x1x14xi1>) -> tensor<29x1x14xi1>
    return %8, %10, %12, %14, %15, %17, %18, %21, %23, %24 : tensor<i1>, tensor<11x99x24x26x66xi1>, tensor<81x27x1xi16>, tensor<1x95x27xi16>, tensor<i1>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<58xi1>, tensor<29x1x14xi1>
  }
}
