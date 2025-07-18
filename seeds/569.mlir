module {
  func.func @main(%arg0: tensor<35x5xi8>, %arg1: tensor<18x86x45xf32>) -> (tensor<18x86x45xf32>, tensor<35x5xi8>) {
    %0 = tosa.identity %arg0 : (tensor<35x5xi8>) -> tensor<35x5xi8>
    %1 = tosa.exp %arg1 : (tensor<18x86x45xf32>) -> tensor<18x86x45xf32>
    %2 = tosa.ceil %1 : (tensor<18x86x45xf32>) -> tensor<18x86x45xf32>
    %3 = tosa.bitwise_or %0, %0 : (tensor<35x5xi8>, tensor<35x5xi8>) -> tensor<35x5xi8>
    %4 = tosa.floor %2 : (tensor<18x86x45xf32>) -> tensor<18x86x45xf32>
    %5 = tosa.bitwise_or %0, %3 : (tensor<35x5xi8>, tensor<35x5xi8>) -> tensor<35x5xi8>
    return %4, %5 : tensor<18x86x45xf32>, tensor<35x5xi8>
  }
}
