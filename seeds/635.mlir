module {
  func.func @main(%arg0: tensor<25x2x62x30x60xi1>, %arg1: tensor<1x1x62x30x1xi1>, %arg2: tensor<42x83xi32>, %arg3: tensor<42x1xi32>, %arg4: tensor<f32>) -> (tensor<25x2x62x30x60xi1>, tensor<42x83xi1>, tensor<f32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<25x2x62x30x60xi1>, tensor<1x1x62x30x1xi1>) -> tensor<25x2x62x30x60xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<25x2x62x30x60xi1>, tensor<25x2x62x30x60xi1>) -> tensor<25x2x62x30x60xi1>
    %2 = tosa.equal %arg2, %arg3 : (tensor<42x83xi32>, tensor<42x1xi32>) -> tensor<42x83xi1>
    %3 = tosa.reciprocal %arg4 : (tensor<f32>) -> tensor<f32>
    return %1, %2, %3 : tensor<25x2x62x30x60xi1>, tensor<42x83xi1>, tensor<f32>
  }
}
