module {
  func.func @main(%arg0: tensor<27x66xi64>, %arg1: tensor<27x66xi64>, %arg2: tensor<f32>, %arg3: tensor<87x57x21x11x27x95xi32>, %arg4: tensor<1x57x1x1x27x1xi32>) -> (tensor<1x66xi1>, tensor<1x66xi1>, tensor<1x1xi1>, tensor<27x66xi1>, tensor<6x9xi1>, tensor<f32>, tensor<1x1xi1>, tensor<f32>, tensor<87x57x21x11x27x95xi32>, tensor<87x57x21x11x27x95xi1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<27x66xi64>, tensor<27x66xi64>) -> tensor<27x66xi1>
    %1 = tosa.abs %0 : (tensor<27x66xi1>) -> tensor<27x66xi1>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<27x66xi1>) -> tensor<1x66xi1>
    %3 = tosa.reduce_min %2 {axis = 1 : i32} : (tensor<1x66xi1>) -> tensor<1x1xi1>
    %4 = tosa.rsqrt %arg2 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.logical_not %3 : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %6 = tosa.pow %4, %4 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = tosa.bitwise_xor %2, %2 : (tensor<1x66xi1>, tensor<1x66xi1>) -> tensor<1x66xi1>
    %8 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<1x66xi1>) -> tensor<1x66xi1>
    %9 = tosa.reverse %5 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %10 = tosa.clz %0 : (tensor<27x66xi1>) -> tensor<27x66xi1>
    %11 = tosa.reciprocal %4 : (tensor<f32>) -> tensor<f32>
    %12 = tosa.abs %11 : (tensor<f32>) -> tensor<f32>
    %13 = tosa.reduce_all %5 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %s_14_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_14_size = tosa.const_shape {values = dense<[ 6, 9 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %14 = tosa.slice %2, %s_14_start, %s_14_size : (tensor<1x66xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<6x9xi1>
    %15 = tosa.abs %12 : (tensor<f32>) -> tensor<f32>
    %16 = tosa.floor %6 : (tensor<f32>) -> tensor<f32>
    %17 = tosa.clz %13 : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %18 = tosa.maximum %arg3, %arg4 : (tensor<87x57x21x11x27x95xi32>, tensor<1x57x1x1x27x1xi32>) -> tensor<87x57x21x11x27x95xi32>
    %19 = tosa.floor %15 : (tensor<f32>) -> tensor<f32>
    %20 = tosa.intdiv %18, %18 : (tensor<87x57x21x11x27x95xi32>, tensor<87x57x21x11x27x95xi32>) -> tensor<87x57x21x11x27x95xi32>
    %21 = tosa.abs %20 : (tensor<87x57x21x11x27x95xi32>) -> tensor<87x57x21x11x27x95xi32>
    %22 = tosa.greater_equal %18, %20 : (tensor<87x57x21x11x27x95xi32>, tensor<87x57x21x11x27x95xi32>) -> tensor<87x57x21x11x27x95xi1>
    return %7, %8, %9, %10, %14, %16, %17, %19, %21, %22 : tensor<1x66xi1>, tensor<1x66xi1>, tensor<1x1xi1>, tensor<27x66xi1>, tensor<6x9xi1>, tensor<f32>, tensor<1x1xi1>, tensor<f32>, tensor<87x57x21x11x27x95xi32>, tensor<87x57x21x11x27x95xi1>
  }
}
