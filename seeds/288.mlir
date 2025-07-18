module {
  func.func @main(%arg0: tensor<12x32x98x51x50xf32>) -> tensor<12x32x98x51x50xf32> {
    %0 = tosa.floor %arg0 : (tensor<12x32x98x51x50xf32>) -> tensor<12x32x98x51x50xf32>
    %1 = tosa.minimum %0, %0 : (tensor<12x32x98x51x50xf32>, tensor<12x32x98x51x50xf32>) -> tensor<12x32x98x51x50xf32>
    return %1 : tensor<12x32x98x51x50xf32>
  }
}
