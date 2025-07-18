module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<78x18x14x68x79x100xi64>) -> (tensor<f32>, tensor<78x18x14x68x79x100xi64>) {
    %0 = tosa.log %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.ceil %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.bitwise_not %arg1 : (tensor<78x18x14x68x79x100xi64>) -> tensor<78x18x14x68x79x100xi64>
    return %1, %2 : tensor<f32>, tensor<78x18x14x68x79x100xi64>
  }
}
