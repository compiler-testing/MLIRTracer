module {
  func.func @main(%arg0: tensor<26x16x6x32x90x7xf32>, %arg1: tensor<47x75xi16>, %arg2: tensor<1x1xi16>) -> (tensor<47x75xi16>, tensor<26x16x6x32x90x7xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<26x16x6x32x90x7xf32>) -> tensor<26x16x6x32x90x7xf32>
    %1 = tosa.logical_right_shift %arg1, %arg2 : (tensor<47x75xi16>, tensor<1x1xi16>) -> tensor<47x75xi16>
    %2 = tosa.sigmoid %0 : (tensor<26x16x6x32x90x7xf32>) -> tensor<26x16x6x32x90x7xf32>
    return %1, %2 : tensor<47x75xi16>, tensor<26x16x6x32x90x7xf32>
  }
}
