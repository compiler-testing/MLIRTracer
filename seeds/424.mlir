module {
  func.func @main(%arg0: tensor<42x6x6x41x92xf32>, %arg1: tensor<42x6x1x41x92xf32>) -> tensor<42x6x6x41x92xf32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<42x6x6x41x92xf32>, tensor<42x6x1x41x92xf32>) -> tensor<42x6x6x41x92xf32>
    return %0 : tensor<42x6x6x41x92xf32>
  }
}
