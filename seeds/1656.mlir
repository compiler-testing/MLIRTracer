module {
  func.func @main(%arg0: tensor<7x63x1x100x37x22xi1>, %arg1: tensor<1x1x1x1x1x22xi1>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<i1>, tensor<7x63x1x100x74x22xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<7x63x1x100x37x22xi1>, tensor<1x1x1x1x1x22xi1>) -> tensor<7x63x1x100x37x22xi1>
    %1 = tosa.sub %0, %0 : (tensor<7x63x1x100x37x22xi1>, tensor<7x63x1x100x37x22xi1>) -> tensor<7x63x1x100x37x22xi1>
    %2 = tosa.greater %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = tosa.concat %1, %1 {axis = 4 : i32} : (tensor<7x63x1x100x37x22xi1>, tensor<7x63x1x100x37x22xi1>) -> tensor<7x63x1x100x74x22xi1>
    return %2, %3 : tensor<i1>, tensor<7x63x1x100x74x22xi1>
  }
}
