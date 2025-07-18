module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<f32>, %arg3: tensor<7x99x60x97x80xi32>, %arg4: tensor<86xf32>, %arg5: tensor<84x29x1xi1>) -> (tensor<i1>, tensor<f32>, tensor<1xf32>, tensor<84x1x1xi1>, tensor<80x97x99x60x7xi32>, tensor<80x97x99x60x7xi32>, tensor<80x97x99x60x7xi32>, tensor<1x1x1xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %2 = tosa.reciprocal %arg2 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.logical_right_shift %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.tanh %2 : (tensor<f32>) -> tensor<f32>
    %6 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %7 = tosa.transpose %arg3 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<7x99x60x97x80xi32>) -> tensor<80x97x99x60x7xi32>
    %8 = tosa.reduce_sum %arg4 {axis = 0 : i32} : (tensor<86xf32>) -> tensor<1xf32>
    %9 = tosa.reduce_all %arg5 {axis = 1 : i32} : (tensor<84x29x1xi1>) -> tensor<84x1x1xi1>
    %10 = tosa.reduce_all %9 {axis = 2 : i32} : (tensor<84x1x1xi1>) -> tensor<84x1x1xi1>
    %11 = tosa.maximum %7, %7 : (tensor<80x97x99x60x7xi32>, tensor<80x97x99x60x7xi32>) -> tensor<80x97x99x60x7xi32>
    %12 = tosa.arithmetic_right_shift %7, %7 {round = true} : (tensor<80x97x99x60x7xi32>, tensor<80x97x99x60x7xi32>) -> tensor<80x97x99x60x7xi32>
    %13 = tosa.bitwise_not %7 : (tensor<80x97x99x60x7xi32>) -> tensor<80x97x99x60x7xi32>
    %14 = tosa.reduce_any %9 {axis = 0 : i32} : (tensor<84x1x1xi1>) -> tensor<1x1x1xi1>
    return %4, %5, %8, %10, %11, %12, %13, %14 : tensor<i1>, tensor<f32>, tensor<1xf32>, tensor<84x1x1xi1>, tensor<80x97x99x60x7xi32>, tensor<80x97x99x60x7xi32>, tensor<80x97x99x60x7xi32>, tensor<1x1x1xi1>
  }
}
