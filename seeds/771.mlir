module {
  func.func @main(%arg0: tensor<92x70x37x15xi8>) -> tensor<92x70x37x15xi1> {
    %0 = tosa.clz %arg0 : (tensor<92x70x37x15xi8>) -> tensor<92x70x37x15xi8>
    %1 = tosa.greater %0, %0 : (tensor<92x70x37x15xi8>, tensor<92x70x37x15xi8>) -> tensor<92x70x37x15xi1>
    %2 = tosa.bitwise_and %1, %1 : (tensor<92x70x37x15xi1>, tensor<92x70x37x15xi1>) -> tensor<92x70x37x15xi1>
    return %2 : tensor<92x70x37x15xi1>
  }
}
