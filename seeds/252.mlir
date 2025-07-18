module {
  func.func @main(%arg0: tensor<98x40x22x1x35xf32>, %arg1: tensor<98x1x1x1x1xf32>, %arg2: tensor<40x62xi32>, %arg3: tensor<40x1xi32>) -> (tensor<98x40x22x1x35xf32>, tensor<1x62xi1>, tensor<40x62xi1>, tensor<40x62xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<98x40x22x1x35xf32>, tensor<98x1x1x1x1xf32>) -> tensor<98x40x22x1x35xf32>
    %1 = tosa.ceil %0 : (tensor<98x40x22x1x35xf32>) -> tensor<98x40x22x1x35xf32>
    %2 = tosa.bitwise_and %arg2, %arg3 : (tensor<40x62xi32>, tensor<40x1xi32>) -> tensor<40x62xi32>
    %3 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<40x62xi32>) -> tensor<1x62xi32>
    %4 = tosa.equal %3, %3 : (tensor<1x62xi32>, tensor<1x62xi32>) -> tensor<1x62xi1>
    %5 = tosa.greater_equal %2, %2 : (tensor<40x62xi32>, tensor<40x62xi32>) -> tensor<40x62xi1>
    %6 = tosa.sub %5, %5 : (tensor<40x62xi1>, tensor<40x62xi1>) -> tensor<40x62xi1>
    %7 = tosa.equal %2, %2 : (tensor<40x62xi32>, tensor<40x62xi32>) -> tensor<40x62xi1>
    return %1, %4, %6, %7 : tensor<98x40x22x1x35xf32>, tensor<1x62xi1>, tensor<40x62xi1>, tensor<40x62xi1>
  }
}
