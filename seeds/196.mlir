module {
  func.func @main(%arg0: tensor<91x33x45xf32>) -> (tensor<91x33x45xf32>, tensor<91x33x45xf32>, tensor<91x33x45xf32>, tensor<2x1xi1>) {
    %0 = tosa.log %arg0 : (tensor<91x33x45xf32>) -> tensor<91x33x45xf32>
    %1 = tosa.argmax %0 {axis = 2 : i32} : (tensor<91x33x45xf32>) -> tensor<91x33xi32>
    %2 = tosa.reciprocal %0 : (tensor<91x33x45xf32>) -> tensor<91x33x45xf32>
    %3 = tosa.exp %2 : (tensor<91x33x45xf32>) -> tensor<91x33x45xf32>
    %4 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<91x33xi32>) -> tensor<1x33xi32>
    %5 = tosa.rsqrt %3 : (tensor<91x33x45xf32>) -> tensor<91x33x45xf32>
    %6 = tosa.equal %4, %4 : (tensor<1x33xi32>, tensor<1x33xi32>) -> tensor<1x33xi1>
    %7 = tosa.reduce_min %6 {axis = 0 : i32} : (tensor<1x33xi1>) -> tensor<1x33xi1>
    %8 = tosa.log %3 : (tensor<91x33x45xf32>) -> tensor<91x33x45xf32>
    %9 = tosa.maximum %3, %3 : (tensor<91x33x45xf32>, tensor<91x33x45xf32>) -> tensor<91x33x45xf32>
    %10 = tosa.bitwise_xor %7, %7 : (tensor<1x33xi1>, tensor<1x33xi1>) -> tensor<1x33xi1>
    %11 = tosa.logical_xor %10, %7 : (tensor<1x33xi1>, tensor<1x33xi1>) -> tensor<1x33xi1>
    %t_12 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %12 = tosa.tile %11, %t_12 : (tensor<1x33xi1>, !tosa.shape<2>) -> tensor<2x66xi1>
    %13 = tosa.logical_right_shift %12, %12 : (tensor<2x66xi1>, tensor<2x66xi1>) -> tensor<2x66xi1>
    %14 = tosa.reduce_min %13 {axis = 1 : i32} : (tensor<2x66xi1>) -> tensor<2x1xi1>
    %15 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %16 = tosa.transpose %14 {perms = array<i32: 0, 1>} : (tensor<2x1xi1>) -> tensor<2x1xi1>
    return %5, %8, %9, %16 : tensor<91x33x45xf32>, tensor<91x33x45xf32>, tensor<91x33x45xf32>, tensor<2x1xi1>
  }
}
