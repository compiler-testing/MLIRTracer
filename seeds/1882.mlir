module {
  func.func @main(%arg0: tensor<42x41x35x26x50x42xi1>, %arg1: tensor<1x41x35x1x50x1xi1>, %arg2: tensor<55x6x6x76x74xf32>, %arg3: tensor<55x1x1x1x74xf32>) -> (tensor<42x41x35x26x50x42xi1>, tensor<55x6x6x76x74xi1>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<42x41x35x26x50x42xi1>, tensor<1x41x35x1x50x1xi1>) -> tensor<42x41x35x26x50x42xi1>
    %1 = tosa.greater_equal %arg2, %arg3 : (tensor<55x6x6x76x74xf32>, tensor<55x1x1x1x74xf32>) -> tensor<55x6x6x76x74xi1>
    return %0, %1 : tensor<42x41x35x26x50x42xi1>, tensor<55x6x6x76x74xi1>
  }
}
