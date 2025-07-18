module {
  func.func @main(%arg0: tensor<94x65x96x72x33x70xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<33x72x65x96x94x70xf32>, tensor<i32>, tensor<i32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<94x65x96x72x33x70xf32>) -> tensor<94x65x96x72x33x70xf32>
    %1 = tosa.bitwise_or %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.sigmoid %0 : (tensor<94x65x96x72x33x70xf32>) -> tensor<94x65x96x72x33x70xf32>
    %3 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %4 = tosa.transpose %2 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<94x65x96x72x33x70xf32>) -> tensor<33x72x65x96x94x70xf32>
    %5 = tosa.clz %1 : (tensor<i32>) -> tensor<i32>
    %6 = tosa.bitwise_and %5, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = tosa.maximum %4, %4 : (tensor<33x72x65x96x94x70xf32>, tensor<33x72x65x96x94x70xf32>) -> tensor<33x72x65x96x94x70xf32>
    %in_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = tosa.negate %6, %in_zp_8, %out_zp_8 : (tensor<i32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    %9 = tosa.clamp %1 {min_val = -38 : i32, max_val = 47 : i32} : (tensor<i32>) -> tensor<i32>
    return %7, %8, %9 : tensor<33x72x65x96x94x70xf32>, tensor<i32>, tensor<i32>
  }
}
