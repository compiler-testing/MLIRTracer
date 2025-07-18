module {
  func.func @main(%arg0: tensor<74x47x84x78x5xi1>, %arg1: tensor<74x47x1x78x5xi1>) -> tensor<74x47x84x78x5xi1> {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<74x47x84x78x5xi1>, tensor<74x47x1x78x5xi1>) -> tensor<74x47x84x78x5xi1>
    return %0 : tensor<74x47x84x78x5xi1>
  }
}
