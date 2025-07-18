module {
  func.func @main(%arg0: tensor<42xf32>, %arg1: tensor<34x85x22xi32>, %arg2: tensor<34x85x1xi32>) -> (tensor<42xi1>, tensor<42xi1>, tensor<42xf32>, tensor<85x34x22xi32>, tensor<42xf32>, tensor<34x85x22xi32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<42xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<42xf32>
    %1 = tosa.exp %0 : (tensor<42xf32>) -> tensor<42xf32>
    %2 = tosa.exp %1 : (tensor<42xf32>) -> tensor<42xf32>
    %3 = tosa.sub %2, %1 : (tensor<42xf32>, tensor<42xf32>) -> tensor<42xf32>
    %4 = tosa.rsqrt %3 : (tensor<42xf32>) -> tensor<42xf32>
    %5 = tosa.greater %4, %4 : (tensor<42xf32>, tensor<42xf32>) -> tensor<42xi1>
    %6 = tosa.greater_equal %0, %1 : (tensor<42xf32>, tensor<42xf32>) -> tensor<42xi1>
    %7 = tosa.intdiv %arg1, %arg2 : (tensor<34x85x22xi32>, tensor<34x85x1xi32>) -> tensor<34x85x22xi32>
    %8 = tosa.sigmoid %0 : (tensor<42xf32>) -> tensor<42xf32>
    %9 = tosa.abs %5 : (tensor<42xi1>) -> tensor<42xi1>
    %10 = tosa.bitwise_not %7 : (tensor<34x85x22xi32>) -> tensor<34x85x22xi32>
    %11 = tosa.rsqrt %3 : (tensor<42xf32>) -> tensor<42xf32>
    %12 = tosa.minimum %10, %10 : (tensor<34x85x22xi32>, tensor<34x85x22xi32>) -> tensor<34x85x22xi32>
    %13 = tosa.arithmetic_right_shift %10, %7 {round = false} : (tensor<34x85x22xi32>, tensor<34x85x22xi32>) -> tensor<34x85x22xi32>
    %14 = tosa.reverse %13 {axis = 0 : i32} : (tensor<34x85x22xi32>) -> tensor<34x85x22xi32>
    %15 = tosa.bitwise_or %12, %7 : (tensor<34x85x22xi32>, tensor<34x85x22xi32>) -> tensor<34x85x22xi32>
    %16 = "tosa.const"() {values = dense<[1, 0, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
    %17 = tosa.transpose %15 {perms = array<i32: 1, 0, 2>} : (tensor<34x85x22xi32>) -> tensor<85x34x22xi32>
    %18 = tosa.pow %8, %2 : (tensor<42xf32>, tensor<42xf32>) -> tensor<42xf32>
    %19 = tosa.maximum %14, %13 : (tensor<34x85x22xi32>, tensor<34x85x22xi32>) -> tensor<34x85x22xi32>
    return %6, %9, %11, %17, %18, %19 : tensor<42xi1>, tensor<42xi1>, tensor<42xf32>, tensor<85x34x22xi32>, tensor<42xf32>, tensor<34x85x22xi32>
  }
}
