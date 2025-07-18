module {
  func.func @main(%arg0: tensor<37x63x70x50xf32>) -> tensor<37x63x70x50xf32> {
    %0 = tosa.abs %arg0 : (tensor<37x63x70x50xf32>) -> tensor<37x63x70x50xf32>
    return %0 : tensor<37x63x70x50xf32>
  }
}
