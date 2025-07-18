module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<82x7x63x47xi16>) -> (tensor<1x7x63x47xi16>, tensor<i1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_sum %arg2 {axis = 0 : i32} : (tensor<82x7x63x47xi16>) -> tensor<1x7x63x47xi16>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<1x7x63x47xi16>) -> tensor<1x7x63x47xi16>
    %3 = tosa.greater %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %2, %3 : tensor<1x7x63x47xi16>, tensor<i1>
  }
}
