module {
  func.func @main(%arg0: tensor<79x28xf32>, %arg1: tensor<25x73x66x7x83x69xi32>) -> (tensor<25x73x66x7x83x69xi32>, tensor<1x28xf32>) {
    %0 = tosa.tanh %arg0 : (tensor<79x28xf32>) -> tensor<79x28xf32>
    %1 = tosa.abs %0 : (tensor<79x28xf32>) -> tensor<79x28xf32>
    %2 = tosa.bitwise_not %arg1 : (tensor<25x73x66x7x83x69xi32>) -> tensor<25x73x66x7x83x69xi32>
    %3 = tosa.clamp %1 {min_val = -6.400000e+01 : f32, max_val = 1.620000e+02 : f32} : (tensor<79x28xf32>) -> tensor<79x28xf32>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<79x28xf32>) -> tensor<1x28xf32>
    return %2, %4 : tensor<25x73x66x7x83x69xi32>, tensor<1x28xf32>
  }
}
