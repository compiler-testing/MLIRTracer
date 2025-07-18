module {
  func.func @main(%arg0: tensor<97x39x54x27x27xi1>, %arg1: tensor<1x1x1x1x1xi1>, %arg2: tensor<f32>, %arg3: tensor<78x91xi1>) -> (tensor<97x39x54x27x27xi1>, tensor<f32>, tensor<78x1xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<97x39x54x27x27xi1>, tensor<1x1x1x1x1xi1>) -> tensor<97x39x54x27x27xi1>
    %1 = tosa.ceil %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.sigmoid %1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.add %2, %1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %4 = tosa.reduce_all %arg3 {axis = 1 : i32} : (tensor<78x91xi1>) -> tensor<78x1xi1>
    return %0, %3, %4 : tensor<97x39x54x27x27xi1>, tensor<f32>, tensor<78x1xi1>
  }
}
