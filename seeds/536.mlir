module {
  func.func @main(%arg0: tensor<91x82x23xi32>, %arg1: tensor<62x89x69x1x7x20xf32>) -> (tensor<1x1xi32>, tensor<1x82x23xi1>, tensor<1x82x23xi32>, tensor<1x1x23xi32>, tensor<7x1x89x69x62x20xf32>, tensor<1xi32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<91x82x23xi32>) -> tensor<1x82x23xi32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<1x82x23xi32>, tensor<1x82x23xi32>) -> tensor<1x82x23xi32>
    %2 = tosa.equal %1, %1 : (tensor<1x82x23xi32>, tensor<1x82x23xi32>) -> tensor<1x82x23xi1>
    %3 = tosa.reciprocal %arg1 : (tensor<62x89x69x1x7x20xf32>) -> tensor<62x89x69x1x7x20xf32>
    %4 = tosa.maximum %3, %3 : (tensor<62x89x69x1x7x20xf32>, tensor<62x89x69x1x7x20xf32>) -> tensor<62x89x69x1x7x20xf32>
    %5 = tosa.tanh %3 : (tensor<62x89x69x1x7x20xf32>) -> tensor<62x89x69x1x7x20xf32>
    %6 = tosa.maximum %4, %3 : (tensor<62x89x69x1x7x20xf32>, tensor<62x89x69x1x7x20xf32>) -> tensor<62x89x69x1x7x20xf32>
    %7 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<1x82x23xi32>) -> tensor<1x1x23xi32>
    %8 = tosa.argmax %7 {axis = 2 : i32} : (tensor<1x1x23xi32>) -> tensor<1x1xi32>
    %9 = tosa.minimum %6, %6 : (tensor<62x89x69x1x7x20xf32>, tensor<62x89x69x1x7x20xf32>) -> tensor<62x89x69x1x7x20xf32>
    %10 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %11 = tosa.transpose %5 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<62x89x69x1x7x20xf32>) -> tensor<7x1x89x69x62x20xf32>
    %12 = tosa.logical_or %2, %2 : (tensor<1x82x23xi1>, tensor<1x82x23xi1>) -> tensor<1x82x23xi1>
    %r_13 = tosa.const_shape {values = dense<[ 110360, 483 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %13 = tosa.reshape %9, %r_13 : (tensor<62x89x69x1x7x20xf32>, !tosa.shape<2>) -> tensor<110360x483xf32>
    %14 = tosa.bitwise_not %0 : (tensor<1x82x23xi32>) -> tensor<1x82x23xi32>
    %15 = tosa.clamp %11 {min_val = 1.200000e+01 : f32, max_val = 6.500000e+01 : f32} : (tensor<7x1x89x69x62x20xf32>) -> tensor<7x1x89x69x62x20xf32>
    %16 = tosa.abs %13 : (tensor<110360x483xf32>) -> tensor<110360x483xf32>
    %17 = tosa.bitwise_xor %7, %7 : (tensor<1x1x23xi32>, tensor<1x1x23xi32>) -> tensor<1x1x23xi32>
    %18 = tosa.argmax %16 {axis = 1 : i32} : (tensor<110360x483xf32>) -> tensor<110360xi32>
    %19 = tosa.reduce_min %18 {axis = 0 : i32} : (tensor<110360xi32>) -> tensor<1xi32>
    %20 = tosa.sub %15, %11 : (tensor<7x1x89x69x62x20xf32>, tensor<7x1x89x69x62x20xf32>) -> tensor<7x1x89x69x62x20xf32>
    %21 = tosa.bitwise_xor %19, %19 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %8, %12, %14, %17, %20, %21 : tensor<1x1xi32>, tensor<1x82x23xi1>, tensor<1x82x23xi32>, tensor<1x1x23xi32>, tensor<7x1x89x69x62x20xf32>, tensor<1xi32>
  }
}
