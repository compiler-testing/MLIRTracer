module {
  func.func @main(%arg0: tensor<61x70xi32>, %arg1: tensor<61x1xi32>, %arg2: tensor<47x74x33x4xi1>, %arg3: tensor<f32>) -> (tensor<47x74x33x4xi1>, tensor<61x70xi32>, tensor<47x148x66x12xi1>, tensor<i1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<61x70xi32>, tensor<61x1xi32>) -> tensor<61x70xi32>
    %1 = tosa.logical_not %arg2 : (tensor<47x74x33x4xi1>) -> tensor<47x74x33x4xi1>
    %2 = tosa.logical_not %1 : (tensor<47x74x33x4xi1>) -> tensor<47x74x33x4xi1>
    %3 = tosa.bitwise_and %2, %1 : (tensor<47x74x33x4xi1>, tensor<47x74x33x4xi1>) -> tensor<47x74x33x4xi1>
    %4 = tosa.add %1, %1 : (tensor<47x74x33x4xi1>, tensor<47x74x33x4xi1>) -> tensor<47x74x33x4xi1>
    %5 = tosa.reciprocal %arg3 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.intdiv %0, %0 : (tensor<61x70xi32>, tensor<61x70xi32>) -> tensor<61x70xi32>
    %t_7 = tosa.const_shape {values = dense<[ 1, 2, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = tosa.tile %4, %t_7 : (tensor<47x74x33x4xi1>, !tosa.shape<4>) -> tensor<47x148x66x12xi1>
    %8 = tosa.greater %5, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %3, %6, %7, %8 : tensor<47x74x33x4xi1>, tensor<61x70xi32>, tensor<47x148x66x12xi1>, tensor<i1>
  }
}
