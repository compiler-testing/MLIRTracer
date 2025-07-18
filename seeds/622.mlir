module {
  func.func @main(%arg0: tensor<61xi1>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<91x77x69x59x90x54xf32>) -> (tensor<61xi1>, tensor<i32>, tensor<91x77x69x59x90x54xf32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<61xi1>) -> tensor<61xi1>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.rsqrt %arg3 : (tensor<91x77x69x59x90x54xf32>) -> tensor<91x77x69x59x90x54xf32>
    %3 = tosa.identity %2 : (tensor<91x77x69x59x90x54xf32>) -> tensor<91x77x69x59x90x54xf32>
    return %0, %1, %3 : tensor<61xi1>, tensor<i32>, tensor<91x77x69x59x90x54xf32>
  }
}
