module {
  func.func @main(%arg0: tensor<26x44x83xf32>, %arg1: tensor<64x87x12xi1>, %arg2: tensor<64x1x1xi1>) -> (tensor<26x83x44xf32>, tensor<66816xi1>, tensor<64x87x12xi1>) {
    %0 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 1>} : (tensor<26x44x83xf32>) -> tensor<26x83x44xf32>
    %2 = tosa.logical_and %arg1, %arg2 : (tensor<64x87x12xi1>, tensor<64x1x1xi1>) -> tensor<64x87x12xi1>
    %3 = tosa.bitwise_and %2, %2 : (tensor<64x87x12xi1>, tensor<64x87x12xi1>) -> tensor<64x87x12xi1>
    %r_4 = tosa.const_shape {values = dense<[ 66816 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.reshape %3, %r_4 : (tensor<64x87x12xi1>, !tosa.shape<1>) -> tensor<66816xi1>
    %5 = tosa.add %2, %2 : (tensor<64x87x12xi1>, tensor<64x87x12xi1>) -> tensor<64x87x12xi1>
    %6 = tosa.bitwise_or %4, %4 : (tensor<66816xi1>, tensor<66816xi1>) -> tensor<66816xi1>
    %7 = tosa.bitwise_or %5, %5 : (tensor<64x87x12xi1>, tensor<64x87x12xi1>) -> tensor<64x87x12xi1>
    return %1, %6, %7 : tensor<26x83x44xf32>, tensor<66816xi1>, tensor<64x87x12xi1>
  }
}
