module {
  func.func @main(%arg0: tensor<64x76x37x42x9x27xi1>, %arg1: tensor<1x76x37x42x9x27xi1>) -> tensor<64x76x37x42x9x27xi1> {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<64x76x37x42x9x27xi1>, tensor<1x76x37x42x9x27xi1>) -> tensor<64x76x37x42x9x27xi1>
    %1 = tosa.clz %0 : (tensor<64x76x37x42x9x27xi1>) -> tensor<64x76x37x42x9x27xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<64x76x37x42x9x27xi1>, tensor<64x76x37x42x9x27xi1>) -> tensor<64x76x37x42x9x27xi1>
    return %2 : tensor<64x76x37x42x9x27xi1>
  }
}
