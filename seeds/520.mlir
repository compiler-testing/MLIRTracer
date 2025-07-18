module {
  func.func @main(%arg0: tensor<65x43xi1>, %arg1: tensor<f32>) -> (tensor<6x129xi1>, tensor<1x1xi1>, tensor<f32>, tensor<i1>, tensor<43xi32>, tensor<i1>, tensor<i1>, tensor<f32>) {
    %0 = tosa.logical_not %arg0 : (tensor<65x43xi1>) -> tensor<65x43xi1>
    %1 = tosa.logical_or %0, %0 : (tensor<65x43xi1>, tensor<65x43xi1>) -> tensor<65x43xi1>
    %2 = tosa.bitwise_or %1, %1 : (tensor<65x43xi1>, tensor<65x43xi1>) -> tensor<65x43xi1>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<65x43xi1>) -> tensor<1x43xi1>
    %4 = tosa.exp %arg1 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.identity %4 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.rsqrt %5 : (tensor<f32>) -> tensor<f32>
    %t_7 = tosa.const_shape {values = dense<[ 3, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.tile %3, %t_7 : (tensor<1x43xi1>, !tosa.shape<2>) -> tensor<3x129xi1>
    %8 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<65x43xi1>) -> tensor<1x43xi1>
    %9 = tosa.bitwise_xor %3, %8 : (tensor<1x43xi1>, tensor<1x43xi1>) -> tensor<1x43xi1>
    %10 = tosa.reciprocal %6 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.exp %5 : (tensor<f32>) -> tensor<f32>
    %12 = tosa.concat %7, %7 {axis = 0 : i32} : (tensor<3x129xi1>, tensor<3x129xi1>) -> tensor<6x129xi1>
    %13 = tosa.log %10 : (tensor<f32>) -> tensor<f32>
    %14 = tosa.reduce_max %9 {axis = 1 : i32} : (tensor<1x43xi1>) -> tensor<1x1xi1>
    %15 = tosa.pow %11, %5 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %16 = tosa.greater %15, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %17 = tosa.logical_not %16 : (tensor<i1>) -> tensor<i1>
    %18 = tosa.bitwise_and %17, %16 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %19 = tosa.log %13 : (tensor<f32>) -> tensor<f32>
    %20 = tosa.logical_xor %16, %18 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %21 = tosa.greater_equal %11, %15 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %22 = tosa.argmax %0 {axis = 0 : i32} : (tensor<65x43xi1>) -> tensor<43xi32>
    %23 = tosa.abs %20 : (tensor<i1>) -> tensor<i1>
    %24 = tosa.logical_and %18, %18 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %25 = tosa.log %15 : (tensor<f32>) -> tensor<f32>
    return %12, %14, %19, %21, %22, %23, %24, %25 : tensor<6x129xi1>, tensor<1x1xi1>, tensor<f32>, tensor<i1>, tensor<43xi32>, tensor<i1>, tensor<i1>, tensor<f32>
  }
}
