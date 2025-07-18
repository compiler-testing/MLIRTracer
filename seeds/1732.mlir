module {
  func.func @main(%arg0: tensor<38x65x64x77x56x28xi8>, %arg1: tensor<25x61x37xf32>) -> (tensor<38x65x64x77x56x28xi1>, tensor<38x65x64x77x56x28xi1>, tensor<25x61x37xf32>) {
    %0 = tosa.identity %arg0 : (tensor<38x65x64x77x56x28xi8>) -> tensor<38x65x64x77x56x28xi8>
    %1 = tosa.greater_equal %0, %0 : (tensor<38x65x64x77x56x28xi8>, tensor<38x65x64x77x56x28xi8>) -> tensor<38x65x64x77x56x28xi1>
    %2 = tosa.greater %0, %0 : (tensor<38x65x64x77x56x28xi8>, tensor<38x65x64x77x56x28xi8>) -> tensor<38x65x64x77x56x28xi1>
    %3 = tosa.bitwise_or %1, %1 : (tensor<38x65x64x77x56x28xi1>, tensor<38x65x64x77x56x28xi1>) -> tensor<38x65x64x77x56x28xi1>
    %4 = tosa.log %arg1 : (tensor<25x61x37xf32>) -> tensor<25x61x37xf32>
    return %2, %3, %4 : tensor<38x65x64x77x56x28xi1>, tensor<38x65x64x77x56x28xi1>, tensor<25x61x37xf32>
  }
}
