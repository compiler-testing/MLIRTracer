module {
  func.func @main(%arg0: tensor<100x82x5x59x33xf32>, %arg1: tensor<1x82x1x1x1xf32>, %arg2: tensor<16x28x71x22x61x78xf32>, %arg3: tensor<87x62xi8>) -> (tensor<16x28x71x22x61x78xf32>, tensor<61x22x28x71x16x78xf32>, tensor<100x82x5x59x33xi1>, tensor<100x82x5x59x33xi1>, tensor<1x62xi8>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<100x82x5x59x33xf32>, tensor<1x82x1x1x1xf32>) -> tensor<100x82x5x59x33xi1>
    %1 = tosa.ceil %arg2 : (tensor<16x28x71x22x61x78xf32>) -> tensor<16x28x71x22x61x78xf32>
    %2 = tosa.identity %1 : (tensor<16x28x71x22x61x78xf32>) -> tensor<16x28x71x22x61x78xf32>
    %3 = tosa.exp %2 : (tensor<16x28x71x22x61x78xf32>) -> tensor<16x28x71x22x61x78xf32>
    %4 = tosa.exp %3 : (tensor<16x28x71x22x61x78xf32>) -> tensor<16x28x71x22x61x78xf32>
    %5 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %6 = tosa.transpose %2 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<16x28x71x22x61x78xf32>) -> tensor<61x22x28x71x16x78xf32>
    %7 = tosa.clz %0 : (tensor<100x82x5x59x33xi1>) -> tensor<100x82x5x59x33xi1>
    %8 = tosa.bitwise_not %0 : (tensor<100x82x5x59x33xi1>) -> tensor<100x82x5x59x33xi1>
    %9 = tosa.reduce_min %arg3 {axis = 0 : i32} : (tensor<87x62xi8>) -> tensor<1x62xi8>
    return %4, %6, %7, %8, %9 : tensor<16x28x71x22x61x78xf32>, tensor<61x22x28x71x16x78xf32>, tensor<100x82x5x59x33xi1>, tensor<100x82x5x59x33xi1>, tensor<1x62xi8>
  }
}
