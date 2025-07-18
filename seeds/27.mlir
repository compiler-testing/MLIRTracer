module {
  func.func @main(%arg0: tensor<21x7x37x8x29x56xf32>, %arg1: tensor<21x7x37x1x29x56xf32>) -> tensor<21x7x37x8x29x56xi1> {
    %0 = tosa.greater %arg0, %arg1 : (tensor<21x7x37x8x29x56xf32>, tensor<21x7x37x1x29x56xf32>) -> tensor<21x7x37x8x29x56xi1>
    %1 = tosa.sub %0, %0 : (tensor<21x7x37x8x29x56xi1>, tensor<21x7x37x8x29x56xi1>) -> tensor<21x7x37x8x29x56xi1>
    return %1 : tensor<21x7x37x8x29x56xi1>
  }
}
