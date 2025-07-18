module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<35x80x45x17x15xi16>) -> (tensor<35x80x45x17x15xi16>, tensor<f32>) {
    %0 = tosa.exp %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.clz %arg1 : (tensor<35x80x45x17x15xi16>) -> tensor<35x80x45x17x15xi16>
    %2 = tosa.bitwise_xor %1, %1 : (tensor<35x80x45x17x15xi16>, tensor<35x80x45x17x15xi16>) -> tensor<35x80x45x17x15xi16>
    %3 = tosa.tanh %0 : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<35x80x45x17x15xi16>, tensor<f32>
  }
}
