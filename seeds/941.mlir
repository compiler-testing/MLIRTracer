module {
  func.func @main(%arg0: tensor<34xi64>, %arg1: tensor<84x72x13x67xf32>, %arg2: tensor<32x64x8x56x17x5xi1>, %arg3: tensor<32x1x1x1x17x5xi1>) -> (tensor<34xi64>, tensor<34xi64>, tensor<32x64x8x56x17x5xi1>, tensor<67x13x72x84xf32>) {
    %0 = tosa.clamp %arg0 {min_val = -4 : i64, max_val = 69 : i64} : (tensor<34xi64>) -> tensor<34xi64>
    %1 = tosa.sub %0, %0 : (tensor<34xi64>, tensor<34xi64>) -> tensor<34xi64>
    %2 = tosa.logical_right_shift %1, %0 : (tensor<34xi64>, tensor<34xi64>) -> tensor<34xi64>
    %3 = tosa.reciprocal %arg1 : (tensor<84x72x13x67xf32>) -> tensor<84x72x13x67xf32>
    %4 = "tosa.const"() {values = dense<[3, 2, 1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
    %5 = tosa.transpose %3 {perms = array<i32: 3, 2, 1, 0>} : (tensor<84x72x13x67xf32>) -> tensor<67x13x72x84xf32>
    %6 = tosa.floor %5 : (tensor<67x13x72x84xf32>) -> tensor<67x13x72x84xf32>
    %7 = tosa.bitwise_not %1 : (tensor<34xi64>) -> tensor<34xi64>
    %8 = tosa.logical_xor %arg2, %arg3 : (tensor<32x64x8x56x17x5xi1>, tensor<32x1x1x1x17x5xi1>) -> tensor<32x64x8x56x17x5xi1>
    %9 = tosa.abs %6 : (tensor<67x13x72x84xf32>) -> tensor<67x13x72x84xf32>
    return %2, %7, %8, %9 : tensor<34xi64>, tensor<34xi64>, tensor<32x64x8x56x17x5xi1>, tensor<67x13x72x84xf32>
  }
}
