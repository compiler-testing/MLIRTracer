module {
  func.func @main(%arg0: tensor<42x63x25x40xi1>, %arg1: tensor<42x1x1x1xi1>) -> tensor<42x63x25x40xi1> {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<42x63x25x40xi1>, tensor<42x1x1x1xi1>) -> tensor<42x63x25x40xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<42x63x25x40xi1>, tensor<42x63x25x40xi1>) -> tensor<42x63x25x40xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<42x63x25x40xi1>, tensor<42x63x25x40xi1>) -> tensor<42x63x25x40xi1>
    return %2 : tensor<42x63x25x40xi1>
  }
}
