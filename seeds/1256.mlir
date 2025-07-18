module {
  func.func @main(%arg0: tensor<40x45x8x89xi1>, %arg1: tensor<43x14xf32>) -> (tensor<40x1x45x1xi1>, tensor<1x14xi1>, tensor<43x14xf32>, tensor<43x28xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<40x45x8x89xi1>) -> tensor<40x45x8x89xi1>
    %1 = tosa.reduce_any %0 {axis = 3 : i32} : (tensor<40x45x8x89xi1>) -> tensor<40x45x8x1xi1>
    %2 = tosa.concat %1, %1 {axis = 3 : i32} : (tensor<40x45x8x1xi1>, tensor<40x45x8x1xi1>) -> tensor<40x45x8x2xi1>
    %3 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %4 = tosa.transpose %2 {perms = array<i32: 0, 3, 1, 2>} : (tensor<40x45x8x2xi1>) -> tensor<40x2x45x8xi1>
    %5 = tosa.reciprocal %arg1 : (tensor<43x14xf32>) -> tensor<43x14xf32>
    %6 = tosa.reduce_all %4 {axis = 3 : i32} : (tensor<40x2x45x8xi1>) -> tensor<40x2x45x1xi1>
    %7 = tosa.bitwise_or %6, %6 : (tensor<40x2x45x1xi1>, tensor<40x2x45x1xi1>) -> tensor<40x2x45x1xi1>
    %8 = tosa.reduce_min %7 {axis = 1 : i32} : (tensor<40x2x45x1xi1>) -> tensor<40x1x45x1xi1>
    %9 = tosa.bitwise_not %8 : (tensor<40x1x45x1xi1>) -> tensor<40x1x45x1xi1>
    %10 = tosa.greater_equal %5, %5 : (tensor<43x14xf32>, tensor<43x14xf32>) -> tensor<43x14xi1>
    %11 = tosa.reduce_all %10 {axis = 0 : i32} : (tensor<43x14xi1>) -> tensor<1x14xi1>
    %12 = tosa.reverse %11 {axis = 1 : i32} : (tensor<1x14xi1>) -> tensor<1x14xi1>
    %13 = tosa.sub %12, %12 : (tensor<1x14xi1>, tensor<1x14xi1>) -> tensor<1x14xi1>
    %14 = tosa.reciprocal %5 : (tensor<43x14xf32>) -> tensor<43x14xf32>
    %t_15 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %15 = tosa.tile %5, %t_15 : (tensor<43x14xf32>, !tosa.shape<2>) -> tensor<43x28xf32>
    return %9, %13, %14, %15 : tensor<40x1x45x1xi1>, tensor<1x14xi1>, tensor<43x14xf32>, tensor<43x28xf32>
  }
}
