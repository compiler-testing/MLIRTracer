module {
  func.func @main(%arg0: tensor<39xf32>, %arg1: tensor<39xf32>, %arg2: tensor<32x83x76xi32>, %arg3: tensor<32x83x76xi32>) -> (tensor<39xf32>, tensor<32x1x76xi1>, tensor<1x83x76xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<39xf32>, tensor<39xf32>) -> tensor<39xf32>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<32x83x76xi32>, tensor<32x83x76xi32>) -> tensor<32x83x76xi32>
    %2 = tosa.greater_equal %1, %1 : (tensor<32x83x76xi32>, tensor<32x83x76xi32>) -> tensor<32x83x76xi1>
    %3 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %4 = tosa.transpose %0 {perms = array<i32: 0>} : (tensor<39xf32>) -> tensor<39xf32>
    %5 = tosa.logical_right_shift %2, %2 : (tensor<32x83x76xi1>, tensor<32x83x76xi1>) -> tensor<32x83x76xi1>
    %6 = tosa.reduce_any %2 {axis = 1 : i32} : (tensor<32x83x76xi1>) -> tensor<32x1x76xi1>
    %7 = tosa.ceil %4 : (tensor<39xf32>) -> tensor<39xf32>
    %8 = tosa.reduce_min %6 {axis = 1 : i32} : (tensor<32x1x76xi1>) -> tensor<32x1x76xi1>
    %9 = tosa.logical_and %8, %6 : (tensor<32x1x76xi1>, tensor<32x1x76xi1>) -> tensor<32x1x76xi1>
    %10 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<32x83x76xi1>) -> tensor<1x83x76xi1>
    return %7, %9, %10 : tensor<39xf32>, tensor<32x1x76xi1>, tensor<1x83x76xi1>
  }
}
